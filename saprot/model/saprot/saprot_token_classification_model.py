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
        
        # 初始化分类头
        if hasattr(self.model, 'embed_tokens'):
            hidden_size = self.model.embed_tokens.weight.shape[1]
        else:
            hidden_size = 2560  # ESM3的标准隐藏维度
            
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels)
        )
    
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
            logits = self.classifier(hidden_states)
        else:
            # 处理不同类型的输入
            if "tokens" in inputs:
                tokens = inputs["tokens"]
                # 确保tokens是float类型而不是long类型，以避免fused_dropout错误
                if tokens.dtype == torch.long:
                    # 创建一个浮点类型的嵌入表示
                    token_embeddings = torch.zeros(tokens.shape[0], tokens.shape[1], hidden_size, 
                                                  device=device, dtype=model_dtype)
                    
                    # 使用ESM3模型进行编码
                    from esm.sdk.api import ESMProtein
                    batch_size = tokens.shape[0]
                    sequence_length = tokens.shape[1]
                    
                    # 创建一个空的输出张量，确保是浮点类型
                    logits = torch.zeros(batch_size, sequence_length, self.num_labels, 
                                        device=device, dtype=model_dtype, requires_grad=True)
                    
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
                                    
                                    # 确保features是浮点类型
                                    features = features.to(dtype=model_dtype)
                                    
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
                    # 确保token_embeddings需要梯度
                    token_embeddings = token_embeddings.detach().requires_grad_(True)
                    logits = self.classifier(token_embeddings)
                else:
                    # 如果已经是浮点类型，直接使用
                    logits = self.classifier(tokens)
                    
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
                
                # 创建一个空的输出张量
                logits = torch.zeros(batch_size, sequence_length, self.num_labels, 
                                    device=device, dtype=model_dtype, requires_grad=True)
                
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
                                
                                # 确保features是浮点类型
                                features = features.to(dtype=model_dtype)
                                
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
                # 确保sequence_embeddings需要梯度
                sequence_embeddings = sequence_embeddings.detach().requires_grad_(True)
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
                # 确保输出需要梯度
                if not outputs[0].requires_grad:
                    outputs = (outputs[0].detach().requires_grad_(True),) + outputs[1:]
                logits = self.classifier(outputs[0])
        
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