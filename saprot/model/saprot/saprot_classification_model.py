import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        # Cache for ESM3 feature dimensions to ensure consistency
        self._feature_dim_cache = None
        self._esm3_encoding_cache = {}
        super().__init__(task="classification", **kwargs)
        
        # 预先创建分类头，使用默认维度，稍后会在first forward时调整
        # 这确保分类头从一开始就在模型参数中
        default_input_dim = 2560  # ESM3的默认输出维度
        self.classification_head = torch.nn.Linear(default_input_dim, self.num_labels)
        print(f"预创建分类头，输入维度: {default_input_dim}, 输出维度: {self.num_labels}")
        
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
        
        # 延迟设置ESM3模型到数据集，因为数据集实例在dataloader创建时才生成
        print("模型setup完成，将在训练开始时设置ESM3模型到数据集")

    def on_train_start(self):
        """训练开始时的回调，确保ESM3模型传递给数据集"""
        super().on_train_start()
        
        # 设置ESM3模型到所有数据集
        self._set_esm_model_to_datasets()

    def on_validation_start(self):
        """验证开始时的回调，确保ESM3模型传递给数据集"""
        super().on_validation_start()
        
        # 设置ESM3模型到所有数据集
        self._set_esm_model_to_datasets()

    def on_test_start(self):
        """测试开始时的回调，确保ESM3模型传递给数据集"""
        super().on_test_start()
        
        # 设置ESM3模型到所有数据集
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

    def forward(self, inputs=None, coords=None, sequences=None, embeddings=None, encoded_proteins=None, **kwargs):
        # Handle different input formats
        if inputs is None and sequences is not None:
            inputs = {"sequences": sequences}
        elif inputs is None and embeddings is not None:
            inputs = {"embeddings": embeddings}
        elif inputs is None and encoded_proteins is not None:
            inputs = {"encoded_proteins": encoded_proteins}
        elif inputs is None:
            inputs = kwargs
        
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # Get device and dtype from model parameters
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # 优先处理encoded_proteins
        if "encoded_proteins" in inputs:
            print(f"[模型调试] 使用encoded_proteins，数量: {len(inputs['encoded_proteins'])}")
            encoded_proteins = inputs["encoded_proteins"]
            
            # 使用ESM3模型的forward方法处理encoded_proteins
            features = []
            for i, encoded_protein in enumerate(encoded_proteins):
                try:
                    with torch.no_grad():
                        # 使用模型的forward方法处理encoded_protein
                        output = self.model.forward(encoded_protein)
                        
                        # 从输出中提取嵌入
                        if hasattr(output, 'embeddings'):
                            embedding = output.embeddings
                        elif hasattr(output, 'last_hidden_state'):
                            embedding = output.last_hidden_state
                        elif hasattr(output, 'sequence_embeddings'):
                            embedding = output.sequence_embeddings
                        else:
                            print(f"[模型调试] encoded_protein {i} 无法找到嵌入，使用零向量")
                            embedding = torch.zeros(2560, device=device, dtype=model_dtype)
                        
                        # 确保正确的设备和数据类型
                        if torch.is_tensor(embedding):
                            embedding = embedding.to(device=device, dtype=model_dtype)
                            # 应用平均池化
                            if embedding.dim() == 3:  # [batch, seq_len, hidden_dim]
                                if embedding.shape[0] == 1:
                                    embedding = embedding.squeeze(0)  # [seq_len, hidden_dim]
                                feature = embedding.mean(dim=0)  # [hidden_dim]
                            elif embedding.dim() == 2:  # [seq_len, hidden_dim]
                                feature = embedding.mean(dim=0)  # [hidden_dim]
                            elif embedding.dim() == 1:  # [hidden_dim]
                                feature = embedding
                            else:
                                feature = embedding.flatten()
                        else:
                            feature = torch.zeros(2560, device=device, dtype=model_dtype)
                        
                        features.append(feature)
                        print(f"[模型调试] encoded_protein {i} 处理完成，特征维度: {feature.shape}")
                        
                except Exception as e:
                    print(f"[模型调试] encoded_protein {i} 处理失败: {str(e)}")
                    feature = torch.zeros(2560, device=device, dtype=model_dtype)
                    features.append(feature)
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, 2560, device=device, dtype=model_dtype)
        
        # 优先处理预编码的嵌入
        elif "embeddings" in inputs:
            print(f"[模型调试] 使用预编码的嵌入，形状: {inputs['embeddings'].shape}")
            stacked_features = inputs["embeddings"].to(device=device, dtype=model_dtype)
        
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
                    
                    # Extract sequence embeddings
                    if hasattr(encoded_protein, 'sequence'):
                        seq_repr = encoded_protein.sequence
                        if torch.is_tensor(seq_repr):
                            # Apply mean pooling if it's a sequence of embeddings
                            if seq_repr.dim() > 1:
                                feature = seq_repr.mean(dim=0)
                            else:
                                feature = seq_repr
                        else:
                            # Convert to tensor with proper device and dtype
                            tensor_repr = torch.tensor(seq_repr, device=device, dtype=model_dtype)
                            if tensor_repr.dim() > 1:
                                feature = tensor_repr.mean(dim=0)
                            else:
                                feature = tensor_repr
                        
                        features.append(feature.to(device=device, dtype=model_dtype))
                        print(f"[模型调试] 序列 {i} 编码完成，特征维度: {feature.shape}")
                    else:
                        print(f"[模型调试] 序列 {i} 编码失败，使用零向量")
                        feature = torch.zeros(2560, device=device, dtype=model_dtype)
                        features.append(feature)
                except Exception as e:
                    print(f"[模型调试] 序列 {i} 编码出错: {str(e)}")
                    feature = torch.zeros(2560, device=device, dtype=model_dtype)
                    features.append(feature)
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, 2560, device=device, dtype=model_dtype)
        
        else:
            print(f"[模型调试] ❌ 输入中没有找到encoded_proteins、embeddings或sequences")
            stacked_features = torch.zeros(1, 2560, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # Get the actual input dimension from the features
        actual_input_dim = stacked_features.shape[-1]
        print(f"[模型调试] 特征维度: {stacked_features.shape}, 分类头输入维度: {self.classification_head.in_features}")
        
        # 检查分类头的输入维度是否匹配，如果不匹配则重建
        if self.classification_head.in_features != actual_input_dim:
            print(f"[模型调试] 🔧 重建分类头: {self.classification_head.in_features} -> {actual_input_dim}")
            self.classification_head = torch.nn.Linear(actual_input_dim, self.num_labels)
            self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
            
            # 更新feature cache
            self._feature_dim_cache = actual_input_dim
            print(f"[模型调试] ✅ 分类头重建完成")
            
            # 重新配置优化器以包含新的分类头参数
            self._reconfigure_optimizer()
        
        # 确保分类头在正确的设备和数据类型上
        self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
        
        # Forward pass
        logits = self.classification_head(stacked_features)
        print(f"[模型调试] 分类输出形状: {logits.shape}")
        
        return logits

    def _reconfigure_optimizer(self):
        """重新配置优化器以包含分类头参数"""
        if hasattr(self, 'trainer') and self.trainer is not None and hasattr(self, 'optimizers'):
            print("重新配置优化器以包含分类头参数")
            
            # 重新初始化优化器以包含新的参数
            self.init_optimizers()
            
            # 如果训练器存在，更新训练器的优化器配置
            if hasattr(self.trainer, 'strategy'):
                self.trainer.strategy.optimizers = [self.optimizer]
                print("优化器重新配置完成")

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
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        # 打印分类头权重信息
        self._print_classification_head_weights("验证")
        
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)

    def _print_classification_head_weights(self, stage_name):
        """打印分类头权重统计信息"""
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            weight = self.classification_head.weight
            bias = self.classification_head.bias
            
            print(f"\n=== {stage_name}阶段结束 - 分类头权重统计 (Epoch {self.current_epoch}) ===")
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