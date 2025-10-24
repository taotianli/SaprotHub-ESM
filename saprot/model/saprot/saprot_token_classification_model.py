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
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        # For MCC calculation
        self.preds = []
        self.targets = []
        super().__init__(task="token_classification", **kwargs)
        
        # 初始化分类头 - 在父类初始化完成后创建
        self.classifier = None
        self._create_classifier()
        
        # 重新初始化优化器以包含分类头参数
        self.init_optimizers()
    
    def _create_classifier(self):
        """创建分类头"""
        # 获取ESM3模型的隐藏维度和数据类型
        if hasattr(self.model, 'embed_tokens'):
            hidden_size = self.model.embed_tokens.weight.shape[1]
        else:
            hidden_size = 2560  # ESM3的标准隐藏维度
        
        # 获取模型的数据类型
        model_dtype = next(self.model.parameters()).dtype
        
        # 创建分类头，确保使用与ESM3模型相同的数据类型
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size, dtype=model_dtype),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, self.num_labels, dtype=model_dtype)
        )
        
        # 确保分类头在正确的设备上
        device = next(self.model.parameters()).device
        self.classifier = self.classifier.to(device=device, dtype=model_dtype)
    
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
        
        # 获取ESM3模型的隐藏维度
        if hasattr(self.model, 'embed_tokens'):
            hidden_size = self.model.embed_tokens.weight.shape[1]
        else:
            hidden_size = 2560  # ESM3的标准隐藏维度
        
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
                logits = self.classifier(token_embeddings)
                    
            elif "embeddings" in inputs:
                embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
                # 确保embeddings需要梯度
                if not embeddings.requires_grad:
                    embeddings = embeddings.detach().requires_grad_(True)
                logits = self.classifier(embeddings)
                
            elif "sequences" in inputs:
                sequences = inputs["sequences"]
                # 使用ESM3模型进行编码
                from esm.sdk.api import ESMProtein
                batch_size = len(sequences)
                sequence_length = max(len(seq) for seq in sequences)
                
                # 创建一个浮点类型的嵌入表示
                sequence_embeddings = torch.zeros(batch_size, sequence_length, hidden_size, 
                                               device=device, dtype=model_dtype)
                
                try:
                    # 对每个序列进行处理
                    for i, seq in enumerate(sequences):
                        protein = ESMProtein(sequence=seq)
                        
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
                                
                                # 存储嵌入表示
                                sequence_embeddings[i] = features
                except Exception as e:
                    print(f"Error processing sequence: {e}")
                
                # 使用分类头处理整个批次的嵌入
                # 确保sequence_embeddings需要梯度且数据类型正确
                sequence_embeddings = sequence_embeddings.detach().requires_grad_(True)
                sequence_embeddings = sequence_embeddings.to(dtype=model_dtype)
                logits = self.classifier(sequence_embeddings)
                    
            else:
                # No valid input format found
                available_keys = list(inputs.keys()) if isinstance(inputs, dict) else "None"
                raise ValueError(
                    f"Input format not recognized. Expected one of: 'tokens', 'embeddings', 'sequences'. "
                    f"Got input keys: {available_keys}"
                )
        
        return logits
    
    def loss_func(self, stage, logits, labels):
        label = labels['labels']
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
        elif lr_scheduler_name == "CosineAnnealingLRScheduler":
            lr_scheduler_cls = CosineAnnealingLRScheduler
        elif lr_scheduler_name == "Esm2LRScheduler":
            lr_scheduler_cls = Esm2LRScheduler
        elif hasattr(torch.optim.lr_scheduler, lr_scheduler_name):
            # 如果是PyTorch内置的调度器
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
        else:
            # 回退到ConstantLRScheduler
            lr_scheduler_cls = ConstantLRScheduler
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)
    
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True):
        """Save checkpoint - only save classifier head and LoRA weights"""
        checkpoint = {
            'classifier_state_dict': self.classifier.state_dict(),
        }
        
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