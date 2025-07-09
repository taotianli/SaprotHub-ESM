import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
# 导入学习率调度器 - 修复导入路径
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, fixed_seq_length: int = 2048, **kwargs):
        """
        Args:
            num_labels: number of labels
            fixed_seq_length: 固定序列长度，用于截断或padding
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        self.fixed_seq_length = fixed_seq_length
        super().__init__(task="classification", **kwargs)
        
        # 创建固定维度的分类头
        self.classification_head = torch.nn.Linear(self.fixed_seq_length, self.num_labels)
        
        # 立即验证分类头是否被正确创建
        # print(f"🔍 立即验证分类头创建...")
        # print(f"分类头存在: {hasattr(self, 'classification_head')}")
        # print(f"分类头不为None: {self.classification_head is not None}")
        
        if self.classification_head is not None:
            param_list = list(self.classification_head.parameters())
            # print(f"分类头参数数量: {len(param_list)}")
            # for i, param in enumerate(param_list):
            #     print(f"  参数 {i}: shape={param.shape}, requires_grad={param.requires_grad}, device={param.device}")
        
        # 确保分类头参数可以训练
        # for name, param in self.classification_head.named_parameters():
        #     print(f"设置参数 {name} 的 requires_grad=True")
        #     param.requires_grad = True
        #     print(f"验证参数 {name}: requires_grad={param.requires_grad}")
            
        # print(f"创建固定分类头: {self.fixed_seq_length} -> {self.num_labels}")
        # print(f"分类头参数: weight={self.classification_head.weight.shape}, bias={self.classification_head.bias.shape}")
        # print(f"分类头参数requires_grad: weight={self.classification_head.weight.requires_grad}, bias={self.classification_head.bias.requires_grad}")
        
        # 重新初始化优化器以包含分类头参数
        # print("重新初始化优化器以包含分类头参数...")
        self.init_optimizers()
        # print("优化器重新初始化完成")
        
    def initialize_metrics(self, stage):
        # For newer versions of torchmetrics, need to specify task type
        if self.num_labels == 2:
            task = "binary"
        else:
            task = "multiclass"
        
        return {f"{stage}_acc": torchmetrics.Accuracy(task=task, num_classes=self.num_labels)}

    def setup(self, stage=None):
        """PyTorch Lightning的setup方法，在这里设置ESM3模型到数据集"""
        super().setup(stage)
        # print("模型setup完成，将在训练开始时设置ESM3模型到数据集")

    def on_train_start(self):
        """训练开始时的回调，确保ESM3模型传递给数据集"""
        super().on_train_start()
        self._set_esm_model_to_datasets()
        
        # 验证分类头参数是否在优化器中
        # self._verify_classification_head_in_optimizer()

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
                # val_dataloaders可能是列表
                if isinstance(val_dl, list):
                    for i, dl in enumerate(val_dl):
                        dataloaders.append((f'val_{i}', dl))
                else:
                    dataloaders.append(('val', val_dl))
                    
            if hasattr(self.trainer, 'test_dataloaders') and self.trainer.test_dataloaders is not None:
                test_dl = self.trainer.test_dataloaders
                if callable(test_dl):
                    test_dl = test_dl()
                # test_dataloaders可能是列表
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

    def forward(self, inputs=None, coords=None, sequences=None, embeddings=None, tokens=None, **kwargs):
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
            # print(f"[模型调试] 使用tokens，形状: {inputs['tokens'].shape}")
            tokens = inputs["tokens"].to(device=device)
            
            # 将tokens转换为浮点数类型并进行截断/padding
            try:
                tokens_float = tokens.float().to(dtype=model_dtype)
                
                if tokens_float.dim() == 2:
                    batch_size, seq_len = tokens_float.shape
                    # print(f"[模型调试] 原始序列长度: {seq_len}, 目标长度: {self.fixed_seq_length}")
                    
                    # 截断或padding到固定长度
                    stacked_features = self._pad_or_truncate_features(tokens_float, self.fixed_seq_length)
                    # print(f"[模型调试] 处理后特征形状: {stacked_features.shape}")
                    
                else:
                    # print(f"[模型调试] ❌ tokens维度不符合预期: {tokens_float.shape}")
                    # 创建固定长度的零特征
                    batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                    stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
                
            except Exception as e:
                # print(f"[模型调试] tokens处理失败: {str(e)}")
                batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs:
            # print(f"[模型调试] 使用预编码的嵌入，形状: {inputs['embeddings'].shape}")
            embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings.dim() == 3:
                # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]
                embeddings = embeddings.mean(dim=2)
            stacked_features = self._pad_or_truncate_features(embeddings, self.fixed_seq_length)
        
        elif "sequences" in inputs:
            # print(f"[模型调试] 处理原始序列，数量: {len(inputs['sequences'])}")
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
                            # print(f"[模型调试] 序列 {i} 编码完成，固定长度: {seq_feature.shape}")
                        else:
                            # print(f"[模型调试] 序列 {i} 编码失败，使用零向量")
                            feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                            features.append(feature)
                    else:
                        # print(f"[模型调试] 序列 {i} 编码失败，使用零向量")
                        feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                        features.append(feature)
                except Exception as e:
                    # print(f"[模型调试] 序列 {i} 编码出错: {str(e)}")
                    feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                    features.append(feature)
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        else:
            # print(f"[模型调试] ❌ 输入中没有找到tokens、embeddings或sequences")
            stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # print(f"[模型调试] 最终特征维度: {stacked_features.shape} (固定长度: {self.fixed_seq_length})")

        # 确保分类头在正确的设备和数据类型上
        self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
        
        # Forward pass
        logits = self.classification_head(stacked_features)
        # print(f"[模型调试] 分类输出形状: {logits.shape}")
        
        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        
        # Compute loss
        loss = cross_entropy(logits, label)
        
        # Update metrics - convert logits to predictions for metric calculation
        # Convert to float32 for metric computation to avoid precision issues
        with torch.no_grad():
            preds = torch.argmax(logits.float(), dim=-1)
            for metric in self.metrics[stage].values():
                metric.update(preds, label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        super().on_train_epoch_end()  # 调用父类方法
        # 打印分类头权重信息
        # self._print_classification_head_weights("训练")

    def on_test_epoch_end(self):
        # 打印分类头权重信息
        # self._print_classification_head_weights("测试")
        
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        # 打印分类头权重信息
        # self._print_classification_head_weights("验证")
        
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)

    def _print_classification_head_weights(self, stage_name):
        """打印分类头权重统计信息"""
        # if hasattr(self, 'classification_head') and self.classification_head is not None:
        #     weight = self.classification_head.weight
        #     bias = self.classification_head.bias
            
        #     print(f"\n=== {stage_name}阶段结束 - 固定分类头权重统计 (Epoch {self.current_epoch}) ===")
        #     print(f"权重矩阵形状: {weight.shape}")
        #     print(f"权重统计: min={weight.min().item():.6f}, max={weight.max().item():.6f}, mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
        #     print(f"权重梯度统计: {'有梯度' if weight.grad is not None else '无梯度'}")
        #     if weight.grad is not None:
        #         print(f"梯度统计: min={weight.grad.min().item():.6f}, max={weight.grad.max().item():.6f}, mean={weight.grad.mean().item():.6f}")
        #         print(f"梯度范数: {weight.grad.norm().item():.6f}")
            
        #     if bias is not None:
        #         print(f"偏置形状: {bias.shape}")
        #         print(f"偏置统计: min={bias.min().item():.6f}, max={bias.max().item():.6f}, mean={bias.mean().item():.6f}")
        #         print(f"偏置梯度统计: {'有梯度' if bias.grad is not None else '无梯度'}")
        #         if bias.grad is not None:
        #             print(f"偏置梯度统计: min={bias.grad.min().item():.6f}, max={bias.grad.max().item():.6f}, mean={bias.grad.mean().item():.6f}")
        #             print(f"偏置梯度范数: {bias.grad.norm().item():.6f}")
            
        #     # 检查权重是否在训练中发生变化
        #     if not hasattr(self, '_prev_weights'):
        #         self._prev_weights = weight.clone().detach()
        #         print("首次记录权重")
        #     else:
        #         weight_diff = torch.abs(weight - self._prev_weights).mean().item()
        #         weight_max_diff = torch.abs(weight - self._prev_weights).max().item()
        #         print(f"权重平均变化量: {weight_diff:.8f}")
        #         print(f"权重最大变化量: {weight_max_diff:.8f}")
        #         if weight_diff < 1e-8:
        #             print("⚠️  警告: 权重几乎没有变化，可能没有在训练!")
        #             # 进一步检查优化器状态
        #             self._check_optimizer_state()
        #         else:
        #             print("✅ 权重正在更新")
        #         self._prev_weights = weight.clone().detach()
            
        #     print("=" * 60 + "\n")
        # else:
        #     print(f"\n⚠️  {stage_name}阶段结束 - 分类头尚未创建 (Epoch {self.current_epoch})\n")
        pass

    def _check_optimizer_state(self):
        """检查优化器状态以诊断训练问题"""
        if hasattr(self, 'optimizer'):
            # print("\n=== 优化器状态诊断 ===")
            
            # 检查学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            # print(f"当前学习率: {current_lr}")
            
            if current_lr == 0:
                # print("❌ 学习率为0，这会阻止参数更新!")
                pass
            elif current_lr < 1e-8:
                # print("⚠️  学习率非常小，可能导致缓慢的收敛")
                pass
            
            # 检查分类头参数是否在优化器中
            classification_head_param_ids = {id(p) for p in self.classification_head.parameters()}
            optimizer_param_ids = set()
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    optimizer_param_ids.add(id(param))
            
            missing_params = classification_head_param_ids - optimizer_param_ids
            if missing_params:
                # print("❌ 分类头参数不在优化器中!")
                pass
            else:
                # print("✅ 分类头参数已在优化器中")
                pass
            
            # 检查梯度
            total_grad_norm = 0.0
            param_count = 0
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
            
            if param_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                # print(f"总梯度范数: {total_grad_norm:.6f}")
                # print(f"有梯度的参数数: {param_count}")
            else:
                # print("❌ 没有参数有梯度!")
                pass
            
            # print("=" * 30)
        pass

    def _verify_classification_head_in_optimizer(self):
        """验证分类头参数是否包含在优化器中"""
        # if hasattr(self, 'classification_head') and hasattr(self, 'optimizer'):
        #     # 获取分类头参数的id
        #     classification_head_param_ids = {id(p) for p in self.classification_head.parameters()}
            
        #     # 获取优化器中所有参数的id
        #     optimizer_param_ids = set()
        #     for param_group in self.optimizer.param_groups:
        #         for param in param_group['params']:
        #             optimizer_param_ids.add(id(param))
            
        #     # 检查交集
        #     included_params = classification_head_param_ids & optimizer_param_ids
        #     missing_params = classification_head_param_ids - optimizer_param_ids
            
        #     print(f"\n=== 分类头参数优化器验证 ===")
        #     print(f"分类头总参数数: {len(classification_head_param_ids)}")
        #     print(f"优化器中的分类头参数数: {len(included_params)}")
        #     print(f"缺失的分类头参数数: {len(missing_params)}")
            
        #     if missing_params:
        #         print("❌ 警告: 以下分类头参数未包含在优化器中:")
        #         for name, param in self.classification_head.named_parameters():
        #             if id(param) in missing_params:
        #                 print(f"  - {name}: {param.shape}, requires_grad={param.requires_grad}")
        #         print("🔧 正在重新初始化优化器...")
        #         self.init_optimizers()
        #         print("✅ 优化器重新初始化完成")
        #     else:
        #         print("✅ 所有分类头参数都已包含在优化器中")
            
        #     # 验证参数的requires_grad设置
        #     print(f"\n=== 分类头参数梯度设置 ===")
        #     for name, param in self.classification_head.named_parameters():
        #         print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}, device={param.device}")
            
        #     print("=" * 50 + "\n")
        pass

    def init_optimizers(self):
        """重写优化器初始化，确保包含分类头参数"""
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
        
        # 添加分类头参数
        classification_head_param_count = 0
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            # print(f"🔍 检查分类头参数...")
            # print(f"分类头类型: {type(self.classification_head)}")
            # print(f"分类头设备: {next(self.classification_head.parameters()).device if list(self.classification_head.parameters()) else 'N/A'}")
            
            for name, param in self.classification_head.named_parameters():
                # print(f"  参数: {name}, shape={param.shape}, requires_grad={param.requires_grad}, device={param.device}")
                if param.requires_grad:
                    full_name = f"classification_head.{name}"
                    all_params.append((full_name, param))
                    classification_head_param_count += 1
                    # print(f"  ✅ 添加到优化器: {full_name}")
                # else:
                #     print(f"  ❌ 跳过（requires_grad=False）: {name}")
        # else:
        #     print("❌ 分类头不存在或为None")

        # print(f"分类头可训练参数数量: {classification_head_param_count}")
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
            
            # print(f"✅ 优化器参数分组:")
            # print(f"  - 带权重衰减的参数: {len(optimizer_grouped_parameters[0]['params'])}")
            # print(f"  - 不带权重衰减的参数: {len(optimizer_grouped_parameters[1]['params'])}")

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
        
        # print(f"✅ 优化器重新初始化完成，总参数组数: {len(optimizer_grouped_parameters)}")
        # print(f"✅ 学习率调度器: {lr_scheduler_name}")
        # print(f"✅ 初始学习率: {self.lr_scheduler_kwargs.get('init_lr', 'N/A')}")

    def training_step(self, batch, batch_idx):
        """重写训练步骤，添加详细的梯度监控"""
        inputs, labels = batch
        
        # 在前向传播前检查参数梯度状态
        # if batch_idx == 0:  # 只在第一个batch时打印
        #     print(f"\n🔍 训练步骤 {batch_idx} 开始前的参数状态:")
        #     for name, param in self.classification_head.named_parameters():
        #         print(f"  {name}: requires_grad={param.requires_grad}, grad={'有' if param.grad is not None else '无'}")
        
        # 前向传播
        outputs = self(**inputs)
        
        # 计算损失
        loss = self.loss_func('train', outputs, labels)
        
        # print(f"🔍 Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        # 在返回loss之前检查梯度（PyTorch Lightning会自动调用backward）
        # if batch_idx == 0:  # 只在第一个batch时打印
        #     print(f"🔍 损失计算完成，准备反向传播...")
        #     print(f"  Loss requires_grad: {loss.requires_grad}")
        #     print(f"  Loss grad_fn: {loss.grad_fn}")
        
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """在优化器步骤之前检查梯度"""
        # 检查分类头梯度
        # total_grad_norm = 0.0
        # param_count = 0
        
        # print(f"\n🔍 优化器步骤前的梯度检查:")
        # for name, param in self.classification_head.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm().item()
        #         total_grad_norm += grad_norm ** 2
        #         param_count += 1
        #         print(f"  {name}: grad_norm={grad_norm:.6f}")
        #     else:
        #         print(f"  {name}: ❌ 无梯度!")
        
        # if param_count > 0:
        #     total_grad_norm = total_grad_norm ** 0.5
        #     print(f"  分类头总梯度范数: {total_grad_norm:.6f}")
        # else:
        #     print(f"  ❌ 分类头没有任何参数有梯度!")
        
        # 调用父类方法
        super().on_before_optimizer_step(optimizer)