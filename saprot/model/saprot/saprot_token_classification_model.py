import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


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
        # For newer versions of torchmetrics, need to specify task type
        return {f"{stage}_acc": torchmetrics.Accuracy(task="multiclass", num_classes=self.num_labels)}
    
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
                                
                                # 存储嵌入表示
                                token_embeddings[i] = features
                except Exception as e:
                    print(f"处理序列时出错: {e}")
                
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
                    print(f"处理序列时出错: {e}")
                
                # 使用分类头处理整个批次的嵌入
                # 确保sequence_embeddings需要梯度且数据类型正确
                sequence_embeddings = sequence_embeddings.detach().requires_grad_(True)
                sequence_embeddings = sequence_embeddings.to(dtype=model_dtype)
                logits = self.classifier(sequence_embeddings)
                    
            else:
                # 使用原有的ESM或BERT模型作为后备
                if hasattr(self.model, "esm"):
                    backbone = self.model.esm
                elif hasattr(self.model, "bert"):
                    backbone = self.model.bert
                else:
                    raise ValueError("模型既没有ESM也没有BERT backbone")
                
                outputs = backbone(**inputs)
                # 确保输出需要梯度且数据类型正确
                hidden_states = outputs[0].to(device=device, dtype=model_dtype)
                if not hidden_states.requires_grad:
                    hidden_states = hidden_states.detach().requires_grad_(True)
                logits = self.classifier(hidden_states)
        
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
        
        # Add the outputs to the list if not in training mode
        if stage != "train":
            preds = logits.argmax(dim=-1)
            self.preds.append(preds)
            self.targets.append(label)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)
        
        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")
        
        return loss
    
    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

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
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

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
            from utils.lr_scheduler import ConstantLRScheduler
            lr_scheduler_cls = ConstantLRScheduler
        elif lr_scheduler_name == "CosineAnnealingLRScheduler":
            from utils.lr_scheduler import CosineAnnealingLRScheduler
            lr_scheduler_cls = CosineAnnealingLRScheduler
        elif lr_scheduler_name == "Esm2LRScheduler":
            from utils.lr_scheduler import Esm2LRScheduler
            lr_scheduler_cls = Esm2LRScheduler
        else:
            # 使用eval来处理其他调度器
            lr_scheduler_cls = eval(lr_scheduler_name)
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)