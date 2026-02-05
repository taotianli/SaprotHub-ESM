import os

# Disable transformers accelerate integration to avoid numpy 2.x compatibility issues
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

import torch
import torch.distributed as dist

# Custom Accuracy implementation to avoid torchmetrics (which triggers accelerate/numpy compatibility issues)
class SimpleAccuracy:
    """Simple accuracy calculation, replaces torchmetrics.Accuracy"""
    def __init__(self, task="multiclass", num_classes=2):
        self.task = task
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0
    
    def update(self, preds, target):
        """Update statistics"""
        self.correct += (preds == target).sum().item()
        self.total += target.numel()
    
    def compute(self):
        """Compute accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        """Reset statistics"""
        self.correct = 0
        self.total = 0

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
# Import learning rate schedulers
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotTokenClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, base_model_type: str = None, **kwargs):
        """
        Args:
            num_labels: number of labels
            base_model_type: 'esm3' or 'esmc', explicitly specify model type
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        self.base_model_type = base_model_type  # 保存base_model_type
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)
        
        # 初始化分类头 - 在父类初始化完成后创建
        self.classifier = None
        self._create_classifier()
        
        # 重新初始化优化器以包含分类头参数
        self.init_optimizers()
    
    def _get_model_hidden_size(self):
        """
        动态检测模型的隐藏维度
        支持 ESM3 不同大小的模型 (open: 2560, 1.4B: 1536) 和 ESMC (960)
        """
        # 方法1: 尝试从 embed_tokens 获取
        if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
            return self.model.embed_tokens.weight.shape[1]
        
        # 方法2: 尝试从 LoRA wrapper 中获取
        if hasattr(self.model, 'esm3_model'):
            inner_model = self.model.esm3_model
            if hasattr(inner_model, 'embed_tokens') and inner_model.embed_tokens is not None:
                return inner_model.embed_tokens.weight.shape[1]
        
        # 方法3: 尝试从 base_model 获取
        if hasattr(self.model, 'base_model'):
            base = self.model.base_model
            if hasattr(base, 'embed_tokens') and base.embed_tokens is not None:
                return base.embed_tokens.weight.shape[1]
        
        # 方法4: 尝试通过一次前向传播获取
        try:
            device = next(self.model.parameters()).device
            # 创建一个简单的测试输入
            test_tokens = torch.tensor([[1, 2, 3]], device=device, dtype=torch.long)
            with torch.no_grad():
                output = self.model.forward(sequence_tokens=test_tokens)
                if hasattr(output, 'embeddings') and output.embeddings is not None:
                    return output.embeddings.shape[-1]
        except Exception:
            pass
        
        # 方法5: 根据 base_model_type 参数判断
        if hasattr(self, 'base_model_type') and self.base_model_type:
            if self.base_model_type == "esmc":
                return 960
            # ESM3 默认使用 2560，但可能是其他变体
        
        # 方法6: 根据模型类名判断
        model_class_name = type(self.model).__name__
        if "ESMC" in model_class_name:
            return 960
        
        # 默认返回 ESM3-1.4B 的隐藏维度
        return 1536
    
    def _create_classifier(self):
        """创建分类头"""
        # 使用统一的方法检测模型隐藏维度
        hidden_size = self._get_model_hidden_size()
        model_info = type(self.model).__name__
        
        # 获取模型的数据类型
        model_dtype = next(self.model.parameters()).dtype
        # print(f"[DEBUG] Model dtype: {model_dtype}")
        
        self.token_hidden_size = hidden_size  # 保存用于forward中的标准化
        
        # 创建简单的分类头：单一线性层
        # 使用普通标准化（在forward中计算），不添加额外的LayerNorm层
        self.classifier = torch.nn.Linear(hidden_size, self.num_labels, dtype=model_dtype)
        
        # 确保分类头在正确的设备上
        device = next(self.model.parameters()).device
        self.classifier = self.classifier.to(device=device, dtype=model_dtype)
        
        # print(f"Token classifier created with hidden_size={hidden_size} for {model_info}")
    
    def compute_mcc(self, preds, target):
        tp = (preds * target).sum()
        tn = ((1 - preds) * (1 - target)).sum()
        fp = (preds * (1 - target)).sum()
        fn = ((1 - preds) * target).sum()
        # Square root each denominator respectively to avoid overflow
        mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
        return tp, tn, fp, fn, mcc
    
    def initialize_metrics(self, stage):
        # 使用自定义的SimpleAccuracy，避免torchmetrics的依赖问题
        return {f"{stage}_acc": SimpleAccuracy(task="multiclass", num_classes=self.num_labels)}
    
    def forward(self, inputs=None, coords=None, sequences=None, embeddings=None, tokens=None, **kwargs):
        # 获取设备和数据类型
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # 使用统一的方法检测模型隐藏维度
        hidden_size = self._get_model_hidden_size()
        
        # 处理不同类型的输入
        if inputs is None and sequences is not None:
            inputs = {"sequences": sequences}
        elif inputs is None and embeddings is not None:
            inputs = {"embeddings": embeddings}
        elif inputs is None and tokens is not None:
            inputs = {"tokens": tokens}
        elif inputs is None:
            inputs = kwargs.get('inputs', {})
        
        # 如果有坐标信息，添加偏置特征
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # 如果backbone被冻结，使用预计算的嵌入
        if self.freeze_backbone:
            hidden_states = self.get_hidden_states_from_dict(inputs, reduction=None)
            if isinstance(hidden_states, list):
                hidden_states = torch.stack(hidden_states)
            # 确保hidden_states的数据类型与模型一致
            hidden_states = hidden_states.to(device=device, dtype=model_dtype)
            # 使用普通标准化
            eps = 1e-6
            feat_mean = hidden_states.mean(dim=-1, keepdim=True)
            feat_std = hidden_states.std(dim=-1, keepdim=True)
            hidden_states = (hidden_states - feat_mean) / (feat_std + eps)
            logits = self.classifier(hidden_states)
        else:
            # 处理不同类型的输入
            if "tokens" in inputs:
                tokens = inputs["tokens"]
                # 确保tokens是long类型用于ESM3输入
                if tokens.dtype != torch.long:
                    tokens = tokens.to(dtype=torch.long)
                
                # 创建一个浮点类型的嵌入表示
                token_embeddings = torch.zeros(tokens.shape[0], tokens.shape[1], hidden_size, 
                                              device=device, dtype=model_dtype)
                
                # 使用ESM3模型进行编码
                from esm.sdk.api import ESMProtein
                batch_size = tokens.shape[0]
                sequence_length = tokens.shape[1]
                
                try:
                    # 对每个序列进行处理
                    for i in range(batch_size):
                        # 确保tokens是long类型用于ESM3输入
                        seq_tokens = tokens[i].to(dtype=torch.long)
                        protein = ESMProtein(tokens=seq_tokens)
                        
                        # 使用no_grad仅用于编码，不用于分类头
                        with torch.no_grad():
                            encoded = self.model.encode(protein)
                            
                        if hasattr(encoded, 'sequence'):
                            seq_features = getattr(encoded, 'sequence')
                            if torch.is_tensor(seq_features):
                                # 确保维度正确
                                if seq_features.dim() == 2:
                                    features = seq_features
                                else:
                                    features = seq_features.unsqueeze(-1).expand(-1, hidden_size)
                                
                                # 确保features是正确的数据类型
                                features = features.to(device=device, dtype=model_dtype)
                                
                                # 截断或padding到正确的长度
                                if len(features) > sequence_length:
                                    features = features[:sequence_length]
                                elif len(features) < sequence_length:
                                    padding = torch.zeros(sequence_length - len(features), hidden_size, 
                                                        device=device, dtype=model_dtype)
                                    features = torch.cat([features, padding])
                                
                                # Store embedding representation
                                token_embeddings[i] = features
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                
                # 使用分类头处理整个批次的嵌入
                # 确保token_embeddings需要梯度且数据类型正确
                token_embeddings = token_embeddings.detach().requires_grad_(True)
                token_embeddings = token_embeddings.to(dtype=model_dtype)
                # 使用普通标准化
                eps = 1e-6
                feat_mean = token_embeddings.mean(dim=-1, keepdim=True)
                feat_std = token_embeddings.std(dim=-1, keepdim=True)
                token_embeddings = (token_embeddings - feat_mean) / (feat_std + eps)
                logits = self.classifier(token_embeddings)
                    
            elif "embeddings" in inputs:
                embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
                # 确保embeddings需要梯度
                if not embeddings.requires_grad:
                    embeddings = embeddings.detach().requires_grad_(True)
                # 使用普通标准化
                eps = 1e-6
                feat_mean = embeddings.mean(dim=-1, keepdim=True)
                feat_std = embeddings.std(dim=-1, keepdim=True)
                embeddings = (embeddings - feat_mean) / (feat_std + eps)
                logits = self.classifier(embeddings)
                
            elif "sequences" in inputs:
                sequences = inputs["sequences"]
                
                # Check if model is ESMC or ESM3
                # 优先使用显式传递的base_model_type参数
                if hasattr(self, 'base_model_type') and self.base_model_type:
                    use_esmc = (self.base_model_type == "esmc")
                else:
                    # 回退到旧的model_type属性
                    model_type = getattr(self, 'model_type', 'esm3')
                    use_esmc = (model_type == "esmc")
                
                # Save sequences for label cleaning if using ESMC
                if use_esmc:
                    self._current_sequences = sequences
                else:
                    self._current_sequences = None
                
                batch_size = len(sequences)
                
                # For ESMC, we need to use cleaned sequence length (without # tokens)
                # For ESM3, we use the original sequence length
                if use_esmc:
                    # Calculate max length of cleaned sequences (without #)
                    sequence_length = max(len(seq.replace('#', '')) for seq in sequences)
                else:
                    sequence_length = max(len(seq) for seq in sequences)
                
                # 创建一个浮点类型的嵌入表示
                sequence_embeddings = torch.zeros(batch_size, sequence_length, hidden_size, 
                                               device=device, dtype=model_dtype)
                
                if use_esmc:
                    # Process sequences using ESMC (using tokenizer approach)
                    try:
                        # Get tokenizer from model
                        tokenizer = self.model.tokenizer
                        
                        # Process each sequence
                        for i, seq in enumerate(sequences):
                            # Remove structure tokens (#) from SA sequence if present
                            # SA sequences have format: "M#T#L#G#R#..." where # are structure tokens
                            # ESMC only needs pure amino acid sequence
                            clean_seq = seq.replace('#', '')
                            
                            # Encode sequence with tokenizer (adds special tokens automatically)
                            tokens = tokenizer.encode(clean_seq, add_special_tokens=True)
                            tokens_tensor = torch.tensor([tokens], device=device)
                            
                            # Use no_grad only for encoding, not for classification head
                            with torch.no_grad():
                                # Forward pass through ESMC
                                out = self.model(tokens_tensor)
                                
                                # Extract embeddings based on output type
                                if hasattr(out, "embeddings"):
                                    token_embs = out.embeddings
                                elif hasattr(out, "last_hidden_state"):
                                    token_embs = out.last_hidden_state
                                elif isinstance(out, dict):
                                    token_embs = list(out.values())[0]
                                elif isinstance(out, (tuple, list)):
                                    token_embs = out[0]
                                else:
                                    raise RuntimeError(f"Cannot extract embeddings from output type: {type(out)}")
                            
                            # Ensure features is 2D [seq_len, hidden_dim]
                            if token_embs.dim() == 3:
                                token_embs = token_embs.squeeze(0)  # Remove batch dimension
                            
                            # Get actual ESMC hidden size from embeddings
                            esmc_hidden_size = token_embs.shape[-1]
                            
                            # Ensure correct dtype
                            features = token_embs.to(device=device, dtype=model_dtype)
                            
                            # Debug: print info once
                            if not hasattr(self, '_esmc_token_debug'):
                                print(f"\n[DEBUG] ESMC Token Info (using tokenizer):")
                                print(f"  Original sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                                print(f"  Original length: {len(seq)}")
                                print(f"  Cleaned sequence: {clean_seq[:50]}{'...' if len(clean_seq) > 50 else ''}")
                                print(f"  Cleaned length: {len(clean_seq)}")
                                print(f"  Encoded tokens length: {len(tokens)}")
                                print(f"  Token embeddings shape: {features.shape}")
                                print(f"  First 10 tokens: {tokens[:10]}")
                                self._esmc_token_debug = True
                            
                            # ESMC tokenizer adds BOS and EOS tokens
                            # Remove them to get only sequence embeddings: features[1:-1]
                            actual_seq_len = len(clean_seq)  # Use cleaned sequence length
                            if len(features) > actual_seq_len:
                                # Skip BOS (first) and EOS (last) tokens
                                features = features[1:-1]
                            
                            # Update sequence_embeddings tensor size if needed
                            if sequence_embeddings.shape[-1] != esmc_hidden_size:
                                sequence_embeddings = torch.zeros(batch_size, sequence_length, esmc_hidden_size,
                                                                device=device, dtype=model_dtype)
                            
                            # Truncate or pad to the target sequence_length (max length in batch)
                            if len(features) > sequence_length:
                                features = features[:sequence_length]
                            elif len(features) < sequence_length:
                                padding = torch.zeros(sequence_length - len(features), esmc_hidden_size, 
                                                    device=device, dtype=model_dtype)
                                features = torch.cat([features, padding], dim=0)
                            
                            # Store embedding representation
                            sequence_embeddings[i] = features
                    except Exception as e:
                        print(f"Error processing ESMC sequence: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Process sequences using ESM3
                    from esm.sdk.api import ESMProtein
                    
                    try:
                        # Process each sequence
                        for i, seq in enumerate(sequences):
                            protein = ESMProtein(sequence=seq)
                            
                            # Use no_grad only for encoding, not for classification head
                            with torch.no_grad():
                                encoded = self.model.encode(protein)
                                
                            if hasattr(encoded, 'sequence'):
                                seq_features = getattr(encoded, 'sequence')
                                if torch.is_tensor(seq_features):
                                    # Ensure correct dimensions
                                    if seq_features.dim() == 2:
                                        features = seq_features
                                    else:
                                        features = seq_features.unsqueeze(-1).expand(-1, hidden_size)
                                    
                                    # Ensure features have correct dtype
                                    features = features.to(device=device, dtype=model_dtype)
                                    
                                    # Truncate or pad to correct length
                                    if len(features) > sequence_length:
                                        features = features[:sequence_length]
                                    elif len(features) < sequence_length:
                                        padding = torch.zeros(sequence_length - len(features), hidden_size, 
                                                            device=device, dtype=model_dtype)
                                        features = torch.cat([features, padding])
                                    
                                    # Store embedding representation
                                    sequence_embeddings[i] = features
                    except Exception as e:
                        print(f"Error processing ESM3 sequence: {e}")
                
                # 使用分类头处理整个批次的嵌入
                # 确保sequence_embeddings需要梯度且数据类型正确
                sequence_embeddings = sequence_embeddings.detach().requires_grad_(True)
                sequence_embeddings = sequence_embeddings.to(dtype=model_dtype)
                # 使用普通标准化
                eps = 1e-6
                feat_mean = sequence_embeddings.mean(dim=-1, keepdim=True)
                feat_std = sequence_embeddings.std(dim=-1, keepdim=True)
                sequence_embeddings = (sequence_embeddings - feat_mean) / (feat_std + eps)
                logits = self.classifier(sequence_embeddings)
                    
            else:
                # No valid input format found
                available_keys = list(inputs.keys()) if isinstance(inputs, dict) else "None"
                raise ValueError(
                    f"Input format not recognized. Expected one of: 'tokens', 'embeddings', 'sequences'. "
                    f"Got input keys: {available_keys}"
                )
        
        # Debug: print shapes before returning (print for first 2 batches)
        # if not hasattr(self, '_forward_debug_count'):
        #     self._forward_debug_count = 0
        # 
        # if self._forward_debug_count < 2:
        #     print(f"\n[DEBUG] Forward output shapes (batch {self._forward_debug_count + 1}):")
        #     print(f"  logits.shape: {logits.shape}")
        #     self._forward_debug_count += 1
        
        return logits
    
    def _clean_labels_for_esmc(self, labels_batch, sequences):
        """
        Clean labels to match ESMC processed sequences.
        Remove labels corresponding to structure tokens (#) in SA sequences.
        
        Args:
            labels_batch: [batch_size, orig_seq_length] - original labels
            sequences: list of original sequences
        Returns:
            cleaned_labels: [batch_size, clean_seq_length] - labels without # positions
        """
        # Debug info
        if not hasattr(self, '_clean_labels_debug'):
            print(f"\n[DEBUG] Cleaning labels for ESMC:")
            print(f"  Original labels_batch shape: {labels_batch.shape}")
            print(f"  Number of sequences: {len(sequences)}")
            if len(sequences) > 0:
                print(f"  First sequence length: {len(sequences[0])}")
                print(f"  First sequence sample: {sequences[0][:50]}...")
            self._clean_labels_debug = True
        
        cleaned_labels_list = []
        for i, (label_seq, orig_seq) in enumerate(zip(labels_batch, sequences)):
            # Find positions without # in the original sequence
            keep_positions = [j for j, char in enumerate(orig_seq) if char != '#']
            
            # Debug first sequence
            if i == 0 and not hasattr(self, '_clean_labels_detail_debug'):
                print(f"  First label_seq length: {len(label_seq)}")
                print(f"  Number of keep positions: {len(keep_positions)}")
                print(f"  Keep positions (first 10): {keep_positions[:10]}")
                self._clean_labels_detail_debug = True
            
            # Extract labels only at those positions
            if len(keep_positions) > 0 and len(keep_positions) <= len(label_seq):
                # Use advanced indexing to extract labels
                keep_positions_tensor = torch.tensor(keep_positions, device=label_seq.device)
                cleaned_label = torch.index_select(label_seq, 0, keep_positions_tensor)
            else:
                cleaned_label = label_seq
            cleaned_labels_list.append(cleaned_label)
        
        # Pad to same length
        max_len = max(len(l) for l in cleaned_labels_list)
        padded_labels = []
        for l in cleaned_labels_list:
            if len(l) < max_len:
                padding = torch.full((max_len - len(l),), -1, dtype=l.dtype, device=l.device)
                l = torch.cat([l, padding])
            padded_labels.append(l)
        
        return torch.stack(padded_labels)
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        
        # Note: Labels are already cleaned in dataset for ESMC models
        # No need to clean them here anymore
        # Just clear the saved sequences if they exist
        if hasattr(self, '_current_sequences') and self._current_sequences is not None:
            self._current_sequences = None
        
        # Debug: print shapes before processing (print for first 2 batches)
        # if not hasattr(self, '_loss_debug_count'):
        #     self._loss_debug_count = 0
        # 
        # if self._loss_debug_count < 2:
        #     print(f"\n[DEBUG] Loss function input shapes (batch {self._loss_debug_count + 1}):")
        #     print(f"  logits.shape: {logits.shape}")
        #     print(f"  label.shape: {label.shape}")
        #     # Move to CPU to safely get min/max
        #     try:
        #         label_cpu = label.cpu()
        #         print(f"  label min/max: {label_cpu.min().item()}/{label_cpu.max().item()}")
        #         print(f"  num_labels: {self.num_labels}")
        #         print(f"  label unique values: {torch.unique(label_cpu).tolist()}")
        #     except Exception as e:
        #         print(f"  Error getting label stats: {e}")
        #     self._loss_debug_count += 1
        
        # Flatten the logits and labels
        logits = logits.view(-1, self.num_labels)
        label = label.view(-1)
        
        # 确保label是long类型
        label = label.to(dtype=torch.long)
        
        # 确保logits需要梯度
        if not logits.requires_grad:
            logits = logits.detach().requires_grad_(True)
            
        loss = cross_entropy(logits, label, ignore_index=-1)
        
        # Remove the ignored index
        mask = label != -1
        label = label[mask]
        logits = logits[mask]
        
        # Get predictions from logits
        preds = logits.argmax(dim=-1)
        
        # Add the outputs to the list if not in training mode
        if stage != "train":
            self.preds.append(preds)
            self.targets.append(label)
        
        # Update metrics with predictions (not logits)
        for metric in self.metrics[stage].values():
            metric.update(preds.detach(), label)
        
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")
        elif stage == "valid":
            # 收集验证损失用于epoch结束时计算平均值
            self.valid_outputs.append(loss.detach())
        elif stage == "test":
            # 收集测试损失用于epoch结束时计算平均值
            self.test_outputs.append(loss.detach())
        
        return loss
    
    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        if len(self.test_outputs) > 0:
            log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))
        else:
            log_dict["test_loss"] = torch.tensor(0.0)

        if len(self.preds) > 0 and len(self.targets) > 0:
            preds = torch.cat(self.preds, dim=-1)
            target = torch.cat(self.targets, dim=-1)
            tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
            
            mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
            log_dict["test_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")
    
    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        if len(self.valid_outputs) > 0:
            log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))
        else:
            log_dict["valid_loss"] = torch.tensor(0.0)

        if len(self.preds) > 0 and len(self.targets) > 0:
            preds = torch.cat(self.preds, dim=-1)
            target = torch.cat(self.targets, dim=-1)
            tp, tn, fp, fn, _ = self.compute_mcc(preds, target)
            
            mcc = (tp * tn - fp * fn) / ((tp + fp).sqrt() * (tp + fn).sqrt() * (tn + fp).sqrt() * (tn + fn).sqrt())
            log_dict["valid_mcc"] = mcc

        # Reset the preds and targets
        self.preds = []
        self.targets = []
        
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)
    
    def init_optimizers(self):
        """重写优化器初始化，确保包含分类头参数"""
        # 检查是否有必要的属性
        if not hasattr(self, 'optimizer_kwargs'):
            # 如果还没有optimizer_kwargs，说明父类初始化还没完成，跳过
            return
            
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
        
        # 添加分类头参数
        classifier_param_count = 0
        if hasattr(self, 'classifier') and self.classifier is not None:
            for name, param in self.classifier.named_parameters():
                if param.requires_grad:
                    full_name = f"classifier.{name}"
                    all_params.append((full_name, param))
                    classifier_param_count += 1
        
        # 按是否需要weight decay分组参数
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in all_params if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in all_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        # 创建优化器 - 使用与abstract_model相同的方式
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
            # ConstantLRScheduler只接受 init_lr 参数，移除其他不支持的参数
            allowed_keys = {'init_lr', 'last_epoch', 'verbose'}
            tmp_kwargs = {k: v for k, v in tmp_kwargs.items() if k in allowed_keys}
        elif lr_scheduler_name == "CosineAnnealingLRScheduler":
            lr_scheduler_cls = CosineAnnealingLRScheduler
            # CosineAnnealingLRScheduler 支持的参数
            allowed_keys = {'init_lr', 'max_lr', 'final_lr', 'warmup_steps', 'cosine_steps', 'last_epoch', 'verbose'}
            tmp_kwargs = {k: v for k, v in tmp_kwargs.items() if k in allowed_keys}
        elif lr_scheduler_name == "Esm2LRScheduler":
            lr_scheduler_cls = Esm2LRScheduler
            # Esm2LRScheduler 支持的参数
            allowed_keys = {'init_lr', 'max_lr', 'final_lr', 'warmup_steps', 'start_decay_after_n_steps', 
                           'end_decay_after_n_steps', 'on_use', 'last_epoch', 'verbose'}
            tmp_kwargs = {k: v for k, v in tmp_kwargs.items() if k in allowed_keys}
        elif hasattr(torch.optim.lr_scheduler, lr_scheduler_name):
            # 如果是PyTorch内置的调度器
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
        else:
            # 回退到ConstantLRScheduler
            lr_scheduler_cls = ConstantLRScheduler
            # 默认使用ConstantLRScheduler，同样过滤参数
            allowed_keys = {'init_lr', 'last_epoch', 'verbose'}
            tmp_kwargs = {k: v for k, v in tmp_kwargs.items() if k in allowed_keys}
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)
    
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True):
        """Save checkpoint - only save classifier head and LoRA weights"""
        checkpoint = {
            'classifier_state_dict': self.classifier.state_dict(),
        }
        
        # Save base_model_type information
        if hasattr(self, 'base_model_type') and self.base_model_type:
            checkpoint['base_model_type'] = self.base_model_type
        
        # Add save_info if provided
        if save_info is not None:
            checkpoint['save_info'] = save_info
        
        # Save LoRA weights if they exist
        if hasattr(self, 'model') and self.model is not None:
            lora_state = {}
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_state[name] = param.data.cpu()
            if lora_state:
                checkpoint['lora_state_dict'] = lora_state
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to: {save_path}")
        print(f"   - Classifier parameters: {len(checkpoint['classifier_state_dict'])}")
        if 'lora_state_dict' in checkpoint:
            print(f"   - LoRA parameters: {len(checkpoint['lora_state_dict'])}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint - only load classifier head and LoRA weights"""
        import os
        
        # Ensure path has .pt extension
        if not path.endswith('.pt'):
            path = path + '.pt'
        
        # Check if path exists
        if not os.path.exists(path):
            print(f"Warning: Checkpoint file not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location='cpu')
        
        # Load classifier head
        if 'classifier_state_dict' in checkpoint:
            self.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print(f"Loaded classifier head, parameters: {len(checkpoint['classifier_state_dict'])}")
        
        # Load LoRA weights
        if 'lora_state_dict' in checkpoint and hasattr(self, 'model'):
            lora_state = checkpoint['lora_state_dict']
            model_dict = dict(self.model.named_parameters())
            loaded_count = 0
            for name, param_data in lora_state.items():
                if name in model_dict:
                    model_dict[name].data.copy_(param_data.to(model_dict[name].device))
                    loaded_count += 1
            print(f"Loaded LoRA weights, parameters: {loaded_count}/{len(lora_state)}")
        
        print(f"Checkpoint loaded successfully: {path}")