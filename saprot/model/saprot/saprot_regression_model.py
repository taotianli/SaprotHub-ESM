import torch.distributed as dist
import torchmetrics
import torch

from ..model_interface import register_model
from .base import SaprotBaseModel
# 导入学习率调度器 - 修复导入路径
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, fixed_seq_length: int = 2048, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            fixed_seq_length: 固定序列长度，用于截断或padding
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        self.fixed_seq_length = fixed_seq_length
        super().__init__(task="regression", **kwargs)
        
        # 创建固定维度的回归头
        self.regression_head = torch.nn.Linear(self.fixed_seq_length, 1)
        
        # print(f"创建固定回归头: {self.fixed_seq_length} -> 1")
        # print(f"回归头参数: weight={self.regression_head.weight.shape}, bias={self.regression_head.bias.shape}")
        
        # 重新初始化优化器以包含回归头参数
        self.init_optimizers()
    
    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}

    def setup(self, stage=None):
        """PyTorch Lightning的setup方法，在这里设置ESM3模型到数据集"""
        super().setup(stage)
        # print("回归模型setup完成，将在训练开始时设置ESM3模型到数据集")

    def on_train_start(self):
        """训练开始时的回调，确保ESM3模型传递给数据集"""
        super().on_train_start()
        self._set_esm_model_to_datasets()

    def on_validation_start(self):
        """验证开始时的回调，确保ESM3模型传递给数据集"""
        super().on_validation_start()
        self._set_esm_model_to_datasets()

    def on_test_start(self):
        """测试开始时的回调，确保ESM3模型传递给数据集"""
        super().on_test_start()
        self._set_esm_model_to_datasets()

    def _set_esm_model_to_datasets(self):
        """将ESM3模型设置到所有数据集"""
        if hasattr(self.trainer, 'datamodule'):
            datasets = []
            
            # 获取所有数据集实例
            if hasattr(self.trainer.datamodule, 'train_dataset'):
                datasets.append(('train', self.trainer.datamodule.train_dataset))
            if hasattr(self.trainer.datamodule, 'val_dataset'):
                datasets.append(('val', self.trainer.datamodule.val_dataset))
            if hasattr(self.trainer.datamodule, 'test_dataset'):
                datasets.append(('test', self.trainer.datamodule.test_dataset))
            
            # 设置ESM3模型
            for stage, dataset in datasets:
                if dataset is not None and hasattr(dataset, 'set_esm_model'):
                    # print(f"设置ESM3模型到{stage}数据集: {type(dataset).__name__}")
                    dataset.set_esm_model(self.model)
                    
            # 另外检查dataloader中的数据集
            dataloaders = []
            
            # 安全地获取dataloaders
            if hasattr(self.trainer, 'train_dataloader') and self.trainer.train_dataloader is not None:
                train_dl = self.trainer.train_dataloader
                if callable(train_dl):
                    train_dl = train_dl()
                dataloaders.append(('train', train_dl))
                
            if hasattr(self.trainer, 'val_dataloaders') and self.trainer.val_dataloaders is not None:
                val_dl = self.trainer.val_dataloaders
                if callable(val_dl):
                    val_dl = val_dl()
                if isinstance(val_dl, list):
                    for i, dl in enumerate(val_dl):
                        dataloaders.append((f'val_{i}', dl))
                else:
                    dataloaders.append(('val', val_dl))
                    
            if hasattr(self.trainer, 'test_dataloaders') and self.trainer.test_dataloaders is not None:
                test_dl = self.trainer.test_dataloaders
                if callable(test_dl):
                    test_dl = test_dl()
                if isinstance(test_dl, list):
                    for i, dl in enumerate(test_dl):
                        dataloaders.append((f'test_{i}', dl))
                else:
                    dataloaders.append(('test', test_dl))
            
            for stage, dataloader in dataloaders:
                if dataloader is not None:
                    if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'set_esm_model'):
                        # print(f"设置ESM3模型到{stage} dataloader数据集: {type(dataloader.dataset).__name__}")
                        dataloader.dataset.set_esm_model(self.model)

    def _pad_or_truncate_features(self, features, target_length):
        """
        将特征截断或padding到固定长度
        Args:
            features: 输入特征 tensor [batch_size, seq_len] 或 [batch_size, seq_len, hidden_dim]
            target_length: 目标长度
        Returns:
            处理后的特征 [batch_size, target_length] 或 [batch_size, target_length, hidden_dim]
        """
        if features.dim() == 2:
            # [batch_size, seq_len] 的情况
            batch_size, seq_len = features.shape
            if seq_len > target_length:
                # 截断
                return features[:, :target_length]
            elif seq_len < target_length:
                # padding
                padding_size = target_length - seq_len
                padding = torch.zeros(batch_size, padding_size, device=features.device, dtype=features.dtype)
                return torch.cat([features, padding], dim=1)
            else:
                return features
        elif features.dim() == 3:
            # [batch_size, seq_len, hidden_dim] 的情况，先平均池化
            features = features.mean(dim=2)  # [batch_size, seq_len]
            return self._pad_or_truncate_features(features, target_length)
        else:
            raise ValueError(f"不支持的特征维度: {features.shape}")
    
    def forward(self, inputs=None, coords=None, sequences=None, embeddings=None, tokens=None, structure_info=None, **kwargs):
        if structure_info:
            # To be implemented
            raise NotImplementedError

        # Handle different input formats
        if inputs is None and sequences is not None:
            inputs = {"sequences": sequences}
        elif inputs is None and embeddings is not None:
            inputs = {"embeddings": embeddings}
        elif inputs is None and tokens is not None:
            inputs = {"tokens": tokens}
        elif inputs is None:
            inputs = kwargs
        
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # Get device and dtype from model parameters
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # 优先处理tokens
        if "tokens" in inputs:
            # print(f"[回归模型调试] 使用tokens，形状: {inputs['tokens'].shape}")
            tokens = inputs["tokens"].to(device=device)
            
            # 将tokens转换为浮点数类型并进行截断/padding
            try:
                tokens_float = tokens.float().to(dtype=model_dtype)
                
                if tokens_float.dim() == 2:
                    batch_size, seq_len = tokens_float.shape
                    # print(f"[回归模型调试] 原始序列长度: {seq_len}, 目标长度: {self.fixed_seq_length}")
                    
                    # 截断或padding到固定长度
                    stacked_features = self._pad_or_truncate_features(tokens_float, self.fixed_seq_length)
                    # print(f"[回归模型调试] 处理后特征形状: {stacked_features.shape}")
                    
                else:
                    # print(f"[回归模型调试] ❌ tokens维度不符合预期: {tokens_float.shape}")
                    # 创建固定长度的零特征
                    batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                    stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
                
            except Exception as e:
                # print(f"[回归模型调试] tokens处理失败: {str(e)}")
                batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs:
            # print(f"[回归模型调试] 使用预编码的嵌入，形状: {inputs['embeddings'].shape}")
            embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings.dim() == 3:
                # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]
                embeddings = embeddings.mean(dim=2)
            stacked_features = self._pad_or_truncate_features(embeddings, self.fixed_seq_length)
        
        elif "sequences" in inputs:
            # print(f"[回归模型调试] 处理原始序列，数量: {len(inputs['sequences'])}")
            sequences = inputs["sequences"]
            
            # Process sequences using ESM3 in the model
            from esm.sdk.api import ESMProtein
            
            features = []
            for i, seq in enumerate(sequences):
                try:
                    protein = ESMProtein(sequence=seq)
                    with torch.no_grad():
                        encoded_protein = self.model.encode(protein)
                    
                    # Extract sequence tokens
                    if hasattr(encoded_protein, 'sequence'):
                        seq_tokens = getattr(encoded_protein, 'sequence')
                        if torch.is_tensor(seq_tokens):
                            # 直接使用tokens作为特征
                            seq_feature = seq_tokens.float()
                            # 截断或padding到固定长度
                            if len(seq_feature) > self.fixed_seq_length:
                                seq_feature = seq_feature[:self.fixed_seq_length]
                            elif len(seq_feature) < self.fixed_seq_length:
                                padding_size = self.fixed_seq_length - len(seq_feature)
                                padding = torch.zeros(padding_size, device=device, dtype=model_dtype)
                                seq_feature = torch.cat([seq_feature, padding])
                            
                            features.append(seq_feature.to(device=device, dtype=model_dtype))
                            # print(f"[回归模型调试] 序列 {i} 编码完成，固定长度: {seq_feature.shape}")
                        else:
                            # print(f"[回归模型调试] 序列 {i} 编码失败，使用零向量")
                            feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                            features.append(feature)
                    else:
                        # print(f"[回归模型调试] 序列 {i} 编码失败，使用零向量")
                        feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                        features.append(feature)
                except Exception as e:
                    # print(f"[回归模型调试] 序列 {i} 编码出错: {str(e)}")
                    feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                    features.append(feature)
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)

        # 保留原有的ESM和ProtBERT逻辑作为兜底
        elif "inputs" in inputs:
            model_inputs = inputs["inputs"]
            
            # For ESM models
            if hasattr(self.model, "esm"):
                # If backbone is frozen, the embedding will be the average of all residues, else it will be the
                # embedding of the <cls> token.
                if self.freeze_backbone:
                    repr = torch.stack(self.get_hidden_states_from_dict(model_inputs, reduction="mean"))
                    x = self.model.classifier.dropout(repr)
                    x = self.model.classifier.dense(x)
                    x = torch.tanh(x)
                    x = self.model.classifier.dropout(x)
                    logits = self.model.classifier.out_proj(x).squeeze(dim=-1)
                else:
                   logits = self.model(**model_inputs).logits.squeeze(dim=-1)
                   
                return logits
        
             # For ProtBERT
            elif hasattr(self.model, "bert"):
                # 检查输入的token IDs是否在有效范围内
                vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
                input_ids = model_inputs["input_ids"]
                if torch.max(input_ids) >= vocab_size:
                    # 将超出范围的ID替换为UNK token ID
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    model_inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
                repr = self.model.bert(**model_inputs).last_hidden_state[:, 0]
                logits = self.model.classifier(repr).squeeze(dim=-1)
                
                return logits
        
        else:
            # print(f"[回归模型调试] ❌ 输入中没有找到tokens、embeddings、sequences或inputs")
            stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # print(f"[回归模型调试] 最终特征维度: {stacked_features.shape} (固定长度: {self.fixed_seq_length})")

        # 确保回归头在正确的设备和数据类型上
        self.regression_head = self.regression_head.to(device=device, dtype=model_dtype)
        
        # Forward pass
        logits = self.regression_head(stacked_features).squeeze(dim=-1)
        # print(f"[回归模型调试] 回归输出形状: {logits.shape}")
        
        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        
        # print("Outputs:", outputs)
        # print("Labels:", fitness)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)
            # print(metric.update(outputs.detach(), fitness))
            # print(metric.compute())

            
        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)
            
            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def init_optimizers(self):
        """重写优化器初始化，确保包含回归头参数"""
        import copy
        copy_optimizer_kwargs = copy.deepcopy(self.optimizer_kwargs)
        
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        weight_decay = copy_optimizer_kwargs.pop("weight_decay")

        # 收集所有需要优化的参数
        all_params = []
        esm3_param_count = 0
        
        # 添加ESM3模型参数
        if hasattr(self, 'model') and self.model is not None:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    all_params.append((name, param))
                    esm3_param_count += 1
        
        # print(f"ESM3模型可训练参数数量: {esm3_param_count}")
        
        # 添加回归头参数
        regression_head_param_count = 0
        if hasattr(self, 'regression_head') and self.regression_head is not None:
            for name, param in self.regression_head.named_parameters():
                if param.requires_grad:
                    full_name = f"regression_head.{name}"
                    all_params.append((full_name, param))
                    regression_head_param_count += 1
                    # print(f"  ✅ 添加到优化器: {full_name}")

        # print(f"回归头可训练参数数量: {regression_head_param_count}")
        # print(f"总可训练参数数量: {len(all_params)}")

        if not all_params:
            # print("⚠️ 警告: 没有找到需要优化的参数!")
            # 创建一个虚拟参数避免优化器错误
            dummy_param = torch.nn.Parameter(torch.tensor(0.0))
            optimizer_grouped_parameters = [
                {'params': [dummy_param], 'weight_decay': 0.0}
            ]
        else:
            # 根据参数名称分组
            optimizer_grouped_parameters = [
                {'params': [param for name, param in all_params if not any(nd in name for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [param for name, param in all_params if any(nd in name for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

        # 创建优化器
        optimizer_cls = eval(f"torch.optim.{copy_optimizer_kwargs.pop('class')}")
        self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                       lr=self.lr_scheduler_kwargs['init_lr'],
                                       **copy_optimizer_kwargs)
        
        # 创建学习率调度器
        tmp_kwargs = copy.deepcopy(self.lr_scheduler_kwargs)
        lr_scheduler_name = tmp_kwargs.pop("class")
        
        # 根据调度器名称选择正确的类
        if lr_scheduler_name == "ConstantLRScheduler":
            lr_scheduler_cls = ConstantLRScheduler
        elif lr_scheduler_name == "CosineAnnealingLRScheduler":
            lr_scheduler_cls = CosineAnnealingLRScheduler
        elif lr_scheduler_name == "Esm2LRScheduler":
            lr_scheduler_cls = Esm2LRScheduler
        elif hasattr(torch.optim.lr_scheduler, lr_scheduler_name):
            # 如果是PyTorch内置的调度器
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
        else:
            # print(f"⚠️  未知的学习率调度器: {lr_scheduler_name}, 使用ConstantLRScheduler")
            lr_scheduler_cls = ConstantLRScheduler
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)

    def training_step(self, batch, batch_idx):
        """重写训练步骤，添加详细的梯度监控"""
        inputs, labels = batch
        
        # 前向传播
        outputs = self(**inputs)
        
        # 计算损失
        loss = self.loss_func('train', outputs, labels)
        
        # print(f"🔍 Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """在优化器步骤之前检查梯度"""
        # 调用父类方法
        super().on_before_optimizer_step(optimizer)

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors
            
            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))
            
            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            if dist.get_rank() == 0:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        # 修复: 支持多元素Tensor，避免.item()报错
                        pred_arr = pred.flatten().tolist()
                        target_arr = target.flatten().tolist()
                        # 如果是单元素，直接写；否则写成逗号分隔
                        pred_str = str(pred_arr[0]) if len(pred_arr) == 1 else ','.join(map(str, pred_arr))
                        target_str = str(target_arr[0]) if len(target_arr) == 1 else ','.join(map(str, target_arr))
                        w.write(f"{pred_str}\t{target_str}\n")
        
        log_dict = self.get_log_dict("test")

        # if dist.get_rank() == 0:
        #     print(log_dict)

        self.output_test_metrics(log_dict)

        self.log_info(log_dict)
        self.reset_metrics("test")

    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        重写保存方法，只保存回归头权重而不是整个ESM3模型
        """
        import os
        import torch
        
        try:
            # 创建保存目录
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 只保存回归头的权重
            if hasattr(self, 'regression_head') and self.regression_head is not None:
                regression_head_state = self.regression_head.state_dict()
                
                # 创建保存的状态字典，只包含回归头
                state_dict = {} if save_info is None else save_info.copy()
                state_dict["regression_head"] = regression_head_state
                state_dict["fixed_seq_length"] = self.fixed_seq_length
                state_dict["task"] = "regression"
                
                # 计算权重文件大小
                param_count = sum(p.numel() for p in self.regression_head.parameters())
                # print(f"🔍 保存回归头权重:")
                # print(f"  - 参数数量: {param_count:,}")
                # print(f"  - 保存路径: {save_path}")
                
                if not save_weights_only:
                    # 如果需要保存训练状态
                    state_dict["global_step"] = self.step
                    state_dict["epoch"] = self.epoch
                    state_dict["best_value"] = getattr(self, "best_value", None)
                    
                    if hasattr(self, 'lr_schedulers') and self.lr_schedulers() is not None:
                        state_dict["lr_scheduler"] = self.lr_schedulers().state_dict()
                    
                    if hasattr(self, 'optimizers') and self.optimizers() is not None:
                        state_dict["optimizer"] = self.optimizers().optimizer.state_dict()
                
                # 保存到文件
                torch.save(state_dict, save_path)
                
                # 验证保存的文件大小
                # saved_size = os.path.getsize(save_path) / (1024 * 1024)
                # print(f"✅ 回归头权重保存成功: {saved_size:.2f} MB")
                
            else:
                print("❌ 回归头不存在，无法保存")
                raise ValueError("Regression head not found")
                
        except Exception as e:
            print(f"❌ 保存回归头权重失败: {str(e)}")
            # 尝试保存到当前目录作为备份
            try:
                fallback_path = os.path.join(os.getcwd(), 'regression_head_checkpoint.pt')
                if hasattr(self, 'regression_head'):
                    state_dict = {"regression_head": self.regression_head.state_dict()}
                    torch.save(state_dict, fallback_path)
                    print(f"💾 备用保存成功: {fallback_path}")
            except Exception as e2:
                print(f"❌ 备用保存也失败: {str(e2)}")
                raise e

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载回归头权重
        """
        import torch
        import os
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 权重文件不存在: {checkpoint_path}")
            return
        
        try:
            # 加载权重
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 验证是否为回归头权重文件
            if "regression_head" in state_dict:
                # 新格式：只包含回归头
                regression_head_state = state_dict["regression_head"]
                fixed_seq_length = state_dict.get("fixed_seq_length", self.fixed_seq_length)
                
                print(f"🔍 加载回归头权重:")
                print(f"  - 文件: {checkpoint_path}")
                print(f"  - 序列长度: {fixed_seq_length}")
                
                # 验证维度匹配
                if fixed_seq_length == self.fixed_seq_length:
                    self.regression_head.load_state_dict(regression_head_state)
                    print(f"✅ 回归头权重加载成功")
                else:
                    print(f"❌ 维度不匹配: 期望({self.fixed_seq_length}, 1), 实际({fixed_seq_length}, 1)")
                    
            elif "model" in state_dict and any("regression_head" in k for k in state_dict["model"].keys()):
                # 旧格式：包含整个模型，提取回归头部分
                model_state = state_dict["model"]
                regression_head_state = {
                    k.replace("regression_head.", ""): v 
                    for k, v in model_state.items() 
                    if k.startswith("regression_head.")
                }
                if regression_head_state:
                    self.regression_head.load_state_dict(regression_head_state)
                    print(f"✅ 从完整模型权重中提取并加载回归头")
                else:
                    print(f"❌ 在模型权重中未找到回归头参数")
            else:
                print(f"❌ 不识别的权重文件格式，state_dict keys: {list(state_dict.keys())}")
                
        except Exception as e:
            print(f"❌ 加载回归头权重失败: {str(e)}")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        
        self.plot_valid_metrics_curve(log_dict)