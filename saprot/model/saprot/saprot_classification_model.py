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
        
    def initialize_metrics(self, stage):
        # For newer versions of torchmetrics, need to specify task type
        if self.num_labels == 2:
            task = "binary"
        else:
            task = "multiclass"
        
        return {f"{stage}_acc": torchmetrics.Accuracy(task=task, num_classes=self.num_labels)}

    def forward(self, inputs=None, coords=None, sequences=None, **kwargs):
        # Handle different input formats
        if inputs is None and sequences is not None:
            inputs = {"sequences": sequences}
        elif inputs is None:
            inputs = kwargs
        
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # Get device and dtype from model parameters
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Handle sequences from ESM3-compatible dataset
        sequences = inputs.get("sequences", None)
        if sequences is not None:
            # Process sequences using ESM3
            from esm.sdk.api import ESMProtein
            
            features = []
            for seq in sequences:
                # Use cached encoding if available
                if seq in self._esm3_encoding_cache:
                    cached_feature = self._esm3_encoding_cache[seq]
                    features.append(cached_feature.to(device=device, dtype=model_dtype))
                    continue
                
                protein = ESMProtein(sequence=seq)
                # Encode protein using ESM3 model
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
                    
                    # Cache the feature (on CPU to save GPU memory)
                    self._esm3_encoding_cache[seq] = feature.cpu()
                    features.append(feature.to(device=device, dtype=model_dtype))
                else:
                    # Determine embedding size if not cached
                    if self._feature_dim_cache is None:
                        try:
                            sample_protein = ESMProtein(sequence="A")
                            sample_encoded = self.model.encode(sample_protein)
                            if hasattr(sample_encoded, 'sequence') and torch.is_tensor(sample_encoded.sequence):
                                if sample_encoded.sequence.dim() > 1:
                                    self._feature_dim_cache = sample_encoded.sequence.shape[-1]
                                else:
                                    self._feature_dim_cache = sample_encoded.sequence.shape[0]
                            else:
                                self._feature_dim_cache = 2560
                        except:
                            self._feature_dim_cache = 2560
                    
                    feature = torch.zeros(self._feature_dim_cache, device=device, dtype=model_dtype)
                    features.append(feature)
            
            # Ensure all features have the same dimension
            if features:
                # Use cached dimension or determine from first feature
                if self._feature_dim_cache is None and len(features) > 0:
                    self._feature_dim_cache = features[0].shape[0]
                
                target_size = self._feature_dim_cache
                normalized_features = []
                
                for feat in features:
                    if feat.shape[0] != target_size:
                        # Resize to target size
                        if feat.shape[0] > target_size:
                            norm_feat = feat[:target_size]
                        else:
                            norm_feat = torch.cat([feat, torch.zeros(target_size - feat.shape[0], device=device, dtype=model_dtype)])
                    else:
                        norm_feat = feat
                    
                    # Ensure proper device and dtype
                    norm_feat = norm_feat.to(device=device, dtype=model_dtype)
                    normalized_features.append(norm_feat)
                
                stacked_features = torch.stack(normalized_features)
            else:
                # Use cached dimension or default
                embedding_size = self._feature_dim_cache if self._feature_dim_cache is not None else 2560
                stacked_features = torch.zeros(1, embedding_size, device=device, dtype=model_dtype)
        
        else:
            # Legacy handling for pre-encoded data
            encoded_proteins = inputs.get("inputs", inputs)
            features = []
            for protein in encoded_proteins:
                seq_attr = getattr(protein, 'sequence', None)
                if seq_attr is not None:
                    if torch.is_tensor(seq_attr):
                        feat = seq_attr.mean(dim=0) if seq_attr.dim() > 1 else seq_attr
                        features.append(feat.to(device=device, dtype=model_dtype))
                    else:
                        tensor_repr = torch.tensor(seq_attr, device=device, dtype=model_dtype)
                        feat = tensor_repr.mean(dim=0) if tensor_repr.dim() > 1 else tensor_repr
                        features.append(feat)
                else:
                    embedding_size = self._feature_dim_cache if self._feature_dim_cache is not None else 2560
                    features.append(torch.zeros(embedding_size, device=device, dtype=model_dtype))
            
            if features:
                stacked_features = torch.stack(features)
            else:
                embedding_size = self._feature_dim_cache if self._feature_dim_cache is not None else 2560
                stacked_features = torch.zeros(1, embedding_size, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # Get the actual input dimension from the features
        actual_input_dim = stacked_features.shape[-1]
        
        # Create classification head only once with consistent dimension
        if not hasattr(self, 'classification_head'):
            self.classification_head = torch.nn.Linear(actual_input_dim, self.num_labels)
            self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
            # Register the classification head as a module
            self.add_module('classification_head', self.classification_head)
            print(f"Created classification head with input dim: {actual_input_dim}")
        elif self.classification_head.in_features != actual_input_dim:
            # This should not happen with proper caching, but add warning
            print(f"Warning: Feature dimension mismatch! Expected {self.classification_head.in_features}, got {actual_input_dim}")
            print("This indicates inconsistent ESM3 encoding. Using zero padding/truncation.")
            
            # Adjust features to match existing classification head
            expected_dim = self.classification_head.in_features
            if actual_input_dim != expected_dim:
                batch_size = stacked_features.shape[0]
                if actual_input_dim > expected_dim:
                    stacked_features = stacked_features[:, :expected_dim]
                else:
                    padding = torch.zeros(batch_size, expected_dim - actual_input_dim, device=device, dtype=model_dtype)
                    stacked_features = torch.cat([stacked_features, padding], dim=1)
        
        # Forward pass
        logits = self.classification_head(stacked_features)
        
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