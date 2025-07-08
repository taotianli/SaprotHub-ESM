import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


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
        print(f"创建固定分类头: {self.fixed_seq_length} -> {self.num_labels}")
        
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
        print("模型setup完成，将在训练开始时设置ESM3模型到数据集")

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
                    print(f"设置ESM3模型到{stage}数据集: {type(dataset).__name__}")
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
                        print(f"设置ESM3模型到{stage} dataloader数据集: {type(dataloader.dataset).__name__}")
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
            print(f"[模型调试] 使用tokens，形状: {inputs['tokens'].shape}")
            tokens = inputs["tokens"].to(device=device)
            
            # 将tokens转换为浮点数类型并进行截断/padding
            try:
                tokens_float = tokens.float().to(dtype=model_dtype)
                
                if tokens_float.dim() == 2:
                    batch_size, seq_len = tokens_float.shape
                    print(f"[模型调试] 原始序列长度: {seq_len}, 目标长度: {self.fixed_seq_length}")
                    
                    # 截断或padding到固定长度
                    stacked_features = self._pad_or_truncate_features(tokens_float, self.fixed_seq_length)
                    print(f"[模型调试] 处理后特征形状: {stacked_features.shape}")
                    
                else:
                    print(f"[模型调试] ❌ tokens维度不符合预期: {tokens_float.shape}")
                    # 创建固定长度的零特征
                    batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                    stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
                
            except Exception as e:
                print(f"[模型调试] tokens处理失败: {str(e)}")
                batch_size = tokens.shape[0] if tokens.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs:
            print(f"[模型调试] 使用预编码的嵌入，形状: {inputs['embeddings'].shape}")
            embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings.dim() == 3:
                # [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len]
                embeddings = embeddings.mean(dim=2)
            stacked_features = self._pad_or_truncate_features(embeddings, self.fixed_seq_length)
        
        elif "sequences" in inputs:
            print(f"[模型调试] 处理原始序列，数量: {len(inputs['sequences'])}")
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
                            print(f"[模型调试] 序列 {i} 编码完成，固定长度: {seq_feature.shape}")
                        else:
                            print(f"[模型调试] 序列 {i} 编码失败，使用零向量")
                            feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                            features.append(feature)
                    else:
                        print(f"[模型调试] 序列 {i} 编码失败，使用零向量")
                        feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                        features.append(feature)
                except Exception as e:
                    print(f"[模型调试] 序列 {i} 编码出错: {str(e)}")
                    feature = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                    features.append(feature)
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        else:
            print(f"[模型调试] ❌ 输入中没有找到tokens、embeddings或sequences")
            stacked_features = torch.zeros(1, self.fixed_seq_length, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        print(f"[模型调试] 最终特征维度: {stacked_features.shape} (固定长度: {self.fixed_seq_length})")

        # 确保分类头在正确的设备和数据类型上
        self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
        
        # Forward pass
        logits = self.classification_head(stacked_features)
        print(f"[模型调试] 分类输出形状: {logits.shape}")
        
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
        self._print_classification_head_weights("训练")

    def on_test_epoch_end(self):
        # 打印分类头权重信息
        self._print_classification_head_weights("测试")
        
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        # 打印分类头权重信息
        self._print_classification_head_weights("验证")
        
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)

    def _print_classification_head_weights(self, stage_name):
        """打印分类头权重统计信息"""
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            weight = self.classification_head.weight
            bias = self.classification_head.bias
            
            print(f"\n=== {stage_name}阶段结束 - 固定分类头权重统计 (Epoch {self.current_epoch}) ===")
            print(f"权重矩阵形状: {weight.shape}")
            print(f"权重统计: min={weight.min().item():.6f}, max={weight.max().item():.6f}, mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
            print(f"权重梯度统计: {'有梯度' if weight.grad is not None else '无梯度'}")
            if weight.grad is not None:
                print(f"梯度统计: min={weight.grad.min().item():.6f}, max={weight.grad.max().item():.6f}, mean={weight.grad.mean().item():.6f}")
            
            if bias is not None:
                print(f"偏置形状: {bias.shape}")
                print(f"偏置统计: min={bias.min().item():.6f}, max={bias.max().item():.6f}, mean={bias.mean().item():.6f}")
                print(f"偏置梯度统计: {'有梯度' if bias.grad is not None else '无梯度'}")
                if bias.grad is not None:
                    print(f"偏置梯度统计: min={bias.grad.min().item():.6f}, max={bias.grad.max().item():.6f}, mean={bias.grad.mean().item():.6f}")
            
            # 检查权重是否在训练中发生变化
            if not hasattr(self, '_prev_weights'):
                self._prev_weights = weight.clone().detach()
                print("首次记录权重")
            else:
                weight_diff = torch.abs(weight - self._prev_weights).mean().item()
                print(f"权重变化量: {weight_diff:.8f}")
                if weight_diff < 1e-8:
                    print("⚠️  警告: 权重几乎没有变化，可能没有在训练!")
                else:
                    print("✓ 权重正在更新")
                self._prev_weights = weight.clone().detach()
            
            print("=" * 60 + "\n")
        else:
            print(f"\n⚠️  {stage_name}阶段结束 - 分类头尚未创建 (Epoch {self.current_epoch})\n")