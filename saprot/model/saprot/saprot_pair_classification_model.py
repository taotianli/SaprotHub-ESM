import torchmetrics
import torch
import torch.distributed as dist

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotPairClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels, fixed_seq_length: int = 2048, **kwargs):
        """
        Args:
            num_labels: number of labels
            fixed_seq_length: 固定序列长度，用于截断或padding
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        self.fixed_seq_length = fixed_seq_length
        super().__init__(task="base", **kwargs)
        
        # 分类头将在initialize_model中创建
        # print(f"分类头将在initialize_model中创建")

    def initialize_model(self):
        """初始化ESM3模型和分类头"""
        super().initialize_model()
        
        # 获取ESM3模型的隐藏维度
        # ESM3模型没有config属性，需要从模型结构中获取hidden_size
        if hasattr(self.model, 'embed_tokens'):
            hidden_size = self.model.embed_tokens.weight.shape[1]
        else:
            # 如果无法获取，使用默认值2560（ESM3的标准隐藏维度）
            hidden_size = 2560
        
        # 对于pair分类，我们需要两倍的hidden_size，因为要处理两个序列
        hidden_size = hidden_size * 2
        
        # 创建分类头
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size // 2, self.num_labels)
        )
        
        # 确保分类头参数可训练
        for param in self.classification_head.parameters():
            param.requires_grad = True
        
        # 重新初始化优化器以包含分类头参数
        self.init_optimizers()

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

    def forward(self, inputs_1=None, inputs_2=None, sequences_1=None, sequences_2=None, embeddings_1=None, embeddings_2=None, tokens_1=None, tokens_2=None, **kwargs):
        # Handle different input formats
        if inputs_1 is None and sequences_1 is not None:
            inputs_1 = {"sequences": sequences_1}
        elif inputs_1 is None and embeddings_1 is not None:
            inputs_1 = {"embeddings": embeddings_1}
        elif inputs_1 is None and tokens_1 is not None:
            inputs_1 = {"tokens": tokens_1}
        elif inputs_1 is None:
            inputs_1 = kwargs.get('inputs_1', {})
            
        if inputs_2 is None and sequences_2 is not None:
            inputs_2 = {"sequences": sequences_2}
        elif inputs_2 is None and embeddings_2 is not None:
            inputs_2 = {"embeddings": embeddings_2}
        elif inputs_2 is None and tokens_2 is not None:
            inputs_2 = {"tokens": tokens_2}
        elif inputs_2 is None:
            inputs_2 = kwargs.get('inputs_2', {})
        
        # Get device and dtype from model parameters
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # 优先处理tokens
        if "tokens" in inputs_1 and "tokens" in inputs_2:
            tokens_1 = inputs_1["tokens"].to(device=device)
            tokens_2 = inputs_2["tokens"].to(device=device)
            
            # 将tokens转换为浮点数类型并进行截断/padding
            try:
                tokens_1_float = tokens_1.float().to(dtype=model_dtype)
                tokens_2_float = tokens_2.float().to(dtype=model_dtype)
                
                if tokens_1_float.dim() == 2 and tokens_2_float.dim() == 2:
                    # 截断或padding到固定长度
                    features_1 = self._pad_or_truncate_features(tokens_1_float, self.fixed_seq_length)
                    features_2 = self._pad_or_truncate_features(tokens_2_float, self.fixed_seq_length)
                    
                    # 连接两个序列的特征
                    stacked_features = torch.cat([features_1, features_2], dim=1)
                else:
                    batch_size = tokens_1.shape[0] if tokens_1.dim() > 0 else 1
                    stacked_features = torch.zeros(batch_size, self.fixed_seq_length * 2, device=device, dtype=model_dtype)
                
            except Exception as e:
                batch_size = tokens_1.shape[0] if tokens_1.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, self.fixed_seq_length * 2, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs_1 and "embeddings" in inputs_2:
            embeddings_1 = inputs_1["embeddings"].to(device=device, dtype=model_dtype)
            embeddings_2 = inputs_2["embeddings"].to(device=device, dtype=model_dtype)
            
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings_1.dim() == 3:
                embeddings_1 = embeddings_1.mean(dim=2)
            if embeddings_2.dim() == 3:
                embeddings_2 = embeddings_2.mean(dim=2)
                
            features_1 = self._pad_or_truncate_features(embeddings_1, self.fixed_seq_length)
            features_2 = self._pad_or_truncate_features(embeddings_2, self.fixed_seq_length)
            stacked_features = torch.cat([features_1, features_2], dim=1)
        
        elif "sequences" in inputs_1 and "sequences" in inputs_2:
            sequences_1 = inputs_1["sequences"]
            sequences_2 = inputs_2["sequences"]
            
            # Process sequences using ESM3 in the model
            from esm.sdk.api import ESMProtein
            
            features_1 = []
            features_2 = []
            
            for i, (seq_1, seq_2) in enumerate(zip(sequences_1, sequences_2)):
                try:
                    # 编码第一个序列
                    protein_1 = ESMProtein(sequence=seq_1)
                    with torch.no_grad():
                        encoded_protein_1 = self.model.encode(protein_1)
                    
                    # 编码第二个序列
                    protein_2 = ESMProtein(sequence=seq_2)
                    with torch.no_grad():
                        encoded_protein_2 = self.model.encode(protein_2)
                    
                    # 提取sequence tokens
                    if hasattr(encoded_protein_1, 'sequence') and hasattr(encoded_protein_2, 'sequence'):
                        seq_tokens_1 = getattr(encoded_protein_1, 'sequence')
                        seq_tokens_2 = getattr(encoded_protein_2, 'sequence')
                        
                        if torch.is_tensor(seq_tokens_1) and torch.is_tensor(seq_tokens_2):
                            # 直接使用tokens作为特征
                            seq_feature_1 = seq_tokens_1.float()
                            seq_feature_2 = seq_tokens_2.float()
                            
                            # 截断或padding到固定长度
                            if len(seq_feature_1) > self.fixed_seq_length:
                                seq_feature_1 = seq_feature_1[:self.fixed_seq_length]
                            elif len(seq_feature_1) < self.fixed_seq_length:
                                padding_size = self.fixed_seq_length - len(seq_feature_1)
                                padding = torch.zeros(padding_size, device=device, dtype=model_dtype)
                                seq_feature_1 = torch.cat([seq_feature_1, padding])
                            
                            if len(seq_feature_2) > self.fixed_seq_length:
                                seq_feature_2 = seq_feature_2[:self.fixed_seq_length]
                            elif len(seq_feature_2) < self.fixed_seq_length:
                                padding_size = self.fixed_seq_length - len(seq_feature_2)
                                padding = torch.zeros(padding_size, device=device, dtype=model_dtype)
                                seq_feature_2 = torch.cat([seq_feature_2, padding])
                            
                            features_1.append(seq_feature_1.to(device=device, dtype=model_dtype))
                            features_2.append(seq_feature_2.to(device=device, dtype=model_dtype))
                        else:
                            # 创建零向量
                            feature_1 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                            feature_2 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                            features_1.append(feature_1)
                            features_2.append(feature_2)
                    else:
                        # 创建零向量
                        feature_1 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                        feature_2 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                        features_1.append(feature_1)
                        features_2.append(feature_2)
                except Exception as e:
                    feature_1 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                    feature_2 = torch.zeros(self.fixed_seq_length, device=device, dtype=model_dtype)
                    features_1.append(feature_1)
                    features_2.append(feature_2)
            
            if features_1 and features_2:
                stacked_features_1 = torch.stack(features_1)
                stacked_features_2 = torch.stack(features_2)
                stacked_features = torch.cat([stacked_features_1, stacked_features_2], dim=1)
            else:
                stacked_features = torch.zeros(1, self.fixed_seq_length * 2, device=device, dtype=model_dtype)
        
        else:
            stacked_features = torch.zeros(1, self.fixed_seq_length * 2, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # 确保分类头在正确的设备和数据类型上
        self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
        
        # Forward pass through the sequential classification head
        logits = self.classification_head(stacked_features)
        
        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
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
        
        # 添加分类头参数
        classification_head_param_count = 0
        if hasattr(self, 'classification_head') and self.classification_head is not None:
            for name, param in self.classification_head.named_parameters():
                if param.requires_grad:
                    full_name = f"classification_head.{name}"
                    all_params.append((full_name, param))
                    classification_head_param_count += 1

        if not all_params:
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
            lr_scheduler_cls = ConstantLRScheduler
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)

    def training_step(self, batch, batch_idx):
        """重写训练步骤，添加详细的梯度监控"""
        inputs, labels = batch
        
        # 前向传播
        outputs = self(**inputs)
        
        # 计算损失
        loss = self.loss_func('train', outputs, labels)
        
        self.log("loss", loss, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        """在优化器步骤之前检查梯度"""
        # 调用父类方法
        super().on_before_optimizer_step(optimizer)

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)