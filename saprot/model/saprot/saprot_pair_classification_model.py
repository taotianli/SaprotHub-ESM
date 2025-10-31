import os

# 禁用transformers的accelerate集成，避免numpy 2.x兼容性问题
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

import torch
import torch.distributed as dist

# 自定义Accuracy实现，避免导入torchmetrics（会触发accelerate/numpy兼容性问题）
class SimpleAccuracy:
    """简单的准确率计算，替代torchmetrics.Accuracy"""
    def __init__(self, task="multiclass", num_classes=2):
        self.task = task
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0
    
    def update(self, preds, target):
        """更新统计"""
        self.correct += (preds == target).sum().item()
        self.total += target.numel()
    
    def compute(self):
        """计算准确率"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        """重置统计"""
        self.correct = 0
        self.total = 0

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotPairClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels, fixed_seq_length: int = 2048, base_model_type: str = None, optimizer_kwargs=None, lr_scheduler_kwargs=None, **kwargs):
        """
        Args:
            num_labels: number of labels
            fixed_seq_length: 固定序列长度，用于截断或padding
            base_model_type: 'esm3' or 'esmc', explicitly specify model type
            optimizer_kwargs: 优化器参数
            lr_scheduler_kwargs: 学习率调度器参数
            **kwargs: other arguments for SaprotBaseModel
        """
        self.base_model_type = base_model_type  # 保存base_model_type
        # 设置优化器和学习率调度器参数
        self.optimizer_kwargs = optimizer_kwargs or {
            "class": "AdamW",
            "weight_decay": 0.01,
            "betas": (0.9, 0.999),
            "eps": 1e-8
        }
        
        self.lr_scheduler_kwargs = lr_scheduler_kwargs or {
            "class": "ConstantLRScheduler",
            "init_lr": 1e-4,
            "num_warmup_steps": 0,
            "num_training_steps": 1000
        }
        
        self.num_labels = num_labels
        self.fixed_seq_length = fixed_seq_length
        super().__init__(task="base", **kwargs)
        
        # 分类头将在initialize_model中创建
        # print(f"分类头将在initialize_model中创建")

    def initialize_model(self):
        """初始化ESM3模型和分类头"""
        super().initialize_model()
        
        # 优先使用显式传递的base_model_type参数
        if hasattr(self, 'base_model_type') and self.base_model_type:
            model_type = self.base_model_type.upper()  # 用于显示
            if self.base_model_type == "esmc":
                hidden_size = 960
            else:  # esm3
                hidden_size = 2560
            # print(f"[DEBUG] Pair Classifier using explicit base_model_type: {self.base_model_type}, hidden_size: {hidden_size}")
        else:
            # 回退到自动检测
            actual_model = self.model
            if hasattr(self.model, 'base_model'):
                actual_model = self.model.base_model
            elif hasattr(self.model, 'esm3_model'):
                actual_model = self.model.esm3_model
            
            model_type = type(actual_model).__name__
            
            if "ESMC" in model_type:
                hidden_size = 960
            elif hasattr(self.model, 'embed_tokens'):
                hidden_size = self.model.embed_tokens.weight.shape[1]
            else:
                hidden_size = 2560
            # print(f"[DEBUG] Pair Classifier auto-detected model_type: {model_type}, hidden_size: {hidden_size}")
        
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
        
        # print(f"Pair classifier created with hidden_size={hidden_size} (single={hidden_size//2}) for {model_type}")
        
        # 重新初始化优化器以包含分类头参数
        self.init_optimizers()

    def initialize_metrics(self, stage):
        # 使用自定义的SimpleAccuracy，避免torchmetrics的依赖问题
        if self.num_labels == 2:
            task = "binary"
        else:
            task = "multiclass"
        
        return {f"{stage}_acc": SimpleAccuracy(task=task, num_classes=self.num_labels)}

    # setup方法已移除，不再需要PyTorch Lightning的setup

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
            # [batch_size, seq_len, hidden_dim] 的情况
            batch_size, seq_len, hidden_dim = features.shape
            if seq_len > target_length:
                # 截断
                return features[:, :target_length, :]
            elif seq_len < target_length:
                # padding
                padding_size = target_length - seq_len
                padding = torch.zeros(batch_size, padding_size, hidden_dim, device=features.device, dtype=features.dtype)
                return torch.cat([features, padding], dim=1)
            else:
                return features
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
        
        # 检测模型类型并获取对应的隐藏维度
        # 优先使用显式传递的base_model_type参数
        if hasattr(self, 'base_model_type') and self.base_model_type:
            if self.base_model_type == "esmc":
                hidden_size = 960
            else:  # esm3
                hidden_size = 2560
        elif hasattr(self.model, 'embed_tokens'):
            hidden_size = self.model.embed_tokens.weight.shape[1]
        else:
            # 回退:尝试从模型类型名称检测
            model_class_name = type(self.model).__name__
            if "ESMC" in model_class_name:
                hidden_size = 960
            else:
                hidden_size = 2560  # ESM3的标准隐藏维度
        
        # 优先处理tokens
        if "tokens" in inputs_1 and "tokens" in inputs_2:
            tokens_1 = inputs_1["tokens"].to(device=device)
            tokens_2 = inputs_2["tokens"].to(device=device)
            
            # 将tokens转换为浮点数类型并进行截断/padding
            try:
                tokens_1_float = tokens_1.float().to(dtype=model_dtype)
                tokens_2_float = tokens_2.float().to(dtype=model_dtype)
                
                if tokens_1_float.dim() == 2 and tokens_2_float.dim() == 2:
                    # 将tokens转换为嵌入维度
                    features_1 = tokens_1_float.unsqueeze(-1).expand(-1, -1, hidden_size)
                    features_2 = tokens_2_float.unsqueeze(-1).expand(-1, -1, hidden_size)
                    
                    # 截断或padding到固定长度
                    features_1 = self._pad_or_truncate_features(features_1, self.fixed_seq_length)
                    features_2 = self._pad_or_truncate_features(features_2, self.fixed_seq_length)
                    
                    # 平均池化得到序列表示
                    features_1 = features_1.mean(dim=1)  # [batch_size, hidden_size]
                    features_2 = features_2.mean(dim=1)  # [batch_size, hidden_size]
                    
                    # 连接两个序列的特征
                    stacked_features = torch.cat([features_1, features_2], dim=1)  # [batch_size, hidden_size*2]
                    # print(f"[DEBUG-TOKENS-BRANCH] Created stacked_features: {stacked_features.shape}")
                else:
                    batch_size = tokens_1.shape[0] if tokens_1.dim() > 0 else 1
                    stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
                    # print(f"[DEBUG-TOKENS-ELSE] Created zero stacked_features: {stacked_features.shape}")
                
            except Exception as e:
                # print(f"[DEBUG-TOKENS-EXCEPTION] Exception in tokens branch: {e}")
                batch_size = tokens_1.shape[0] if tokens_1.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
                # print(f"[DEBUG-TOKENS-EXCEPTION] Created zero stacked_features: {stacked_features.shape}")
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs_1 and "embeddings" in inputs_2:
            # print(f"[DEBUG-EMBEDDINGS] Using embeddings branch")
            embeddings_1 = inputs_1["embeddings"].to(device=device, dtype=model_dtype)
            embeddings_2 = inputs_2["embeddings"].to(device=device, dtype=model_dtype)
            
            # Debug: print shapes once
            # if not hasattr(self, '_embeddings_debug'):
            #     print(f"\n[DEBUG] Embeddings input:")
            #     print(f"  embeddings_1.shape: {embeddings_1.shape}")
            #     print(f"  embeddings_2.shape: {embeddings_2.shape}")
            #     print(f"  Expected hidden_size: {hidden_size}")
            #     self._embeddings_debug = True
            
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings_1.dim() == 3:
                embeddings_1 = embeddings_1.mean(dim=1)  # [batch_size, hidden_size]
            if embeddings_2.dim() == 3:
                embeddings_2 = embeddings_2.mean(dim=1)  # [batch_size, hidden_size]
            
            # 确保 embeddings 的最后一个维度与 hidden_size 匹配
            # 如果不匹配,需要进行投影或调整
            if embeddings_1.shape[-1] != hidden_size:
                # print(f"[WARNING] embeddings_1 size mismatch: {embeddings_1.shape[-1]} vs expected {hidden_size}")
                # 如果维度不匹配,创建零向量
                batch_size = embeddings_1.shape[0]
                embeddings_1 = torch.zeros(batch_size, hidden_size, device=device, dtype=model_dtype)
            if embeddings_2.shape[-1] != hidden_size:
                # print(f"[WARNING] embeddings_2 size mismatch: {embeddings_2.shape[-1]} vs expected {hidden_size}")
                batch_size = embeddings_2.shape[0]
                embeddings_2 = torch.zeros(batch_size, hidden_size, device=device, dtype=model_dtype)
                
            stacked_features = torch.cat([embeddings_1, embeddings_2], dim=1)  # [batch_size, hidden_size*2]
            # print(f"[DEBUG-EMBEDDINGS] Created stacked_features: {stacked_features.shape}")
        
        elif "sequences" in inputs_1 and "sequences" in inputs_2:
            # print(f"[DEBUG-SEQUENCES] Using sequences branch")
            sequences_1 = inputs_1["sequences"]
            sequences_2 = inputs_2["sequences"]
            
            # Check if model is ESMC or ESM3
            # 优先使用显式传递的base_model_type参数
            if hasattr(self, 'base_model_type') and self.base_model_type:
                use_esmc = (self.base_model_type == "esmc")
            else:
                # 回退到检测模型类型
                model_class_name = type(self.model).__name__
                use_esmc = ("ESMC" in model_class_name)
            
            features_1 = []
            features_2 = []
            
            if use_esmc:
                # Process sequences using ESMC
                from esm.sdk.api import ESMProtein, LogitsConfig
                
                # Get actual ESMC hidden size from first sequence
                esmc_hidden_size = None
                
                for i, (seq_1, seq_2) in enumerate(zip(sequences_1, sequences_2)):
                    try:
                        # Encode first sequence
                        protein_1 = ESMProtein(sequence=seq_1)
                        with torch.no_grad():
                            protein_tensor_1 = self.model.encode(protein_1)
                            logits_output_1 = self.model.logits(
                                protein_tensor_1, LogitsConfig(sequence=True, return_embeddings=True)
                            )
                        
                        # Encode second sequence
                        protein_2 = ESMProtein(sequence=seq_2)
                        with torch.no_grad():
                            protein_tensor_2 = self.model.encode(protein_2)
                            logits_output_2 = self.model.logits(
                                protein_tensor_2, LogitsConfig(sequence=True, return_embeddings=True)
                            )
                        
                        if hasattr(logits_output_1, 'embeddings') and logits_output_1.embeddings is not None and \
                           hasattr(logits_output_2, 'embeddings') and logits_output_2.embeddings is not None:
                            # embeddings shape: [seq_len, hidden_dim]
                            # Get actual hidden size from embeddings
                            if esmc_hidden_size is None:
                                esmc_hidden_size = logits_output_1.embeddings.shape[-1]
                                # print(f"[DEBUG-ESMC-EMBED] embeddings_1 shape: {logits_output_1.embeddings.shape}")
                                # print(f"[DEBUG-ESMC-EMBED] embeddings_2 shape: {logits_output_2.embeddings.shape}")
                            
                            # 如果embeddings是3D [batch, seq_len, hidden_dim],先去掉batch维度
                            emb_1 = logits_output_1.embeddings
                            emb_2 = logits_output_2.embeddings
                            if emb_1.dim() == 3:
                                emb_1 = emb_1.squeeze(0)  # [seq_len, hidden_dim]
                            if emb_2.dim() == 3:
                                emb_2 = emb_2.squeeze(0)  # [seq_len, hidden_dim]
                            
                            seq_feature_1 = emb_1.mean(dim=0).float()  # [esmc_hidden_size]
                            seq_feature_2 = emb_2.mean(dim=0).float()  # [esmc_hidden_size]
                            
                            # if esmc_hidden_size is None or i == 0:
                            #     print(f"[DEBUG-ESMC-POOLED] seq_feature_1 shape: {seq_feature_1.shape}")
                            #     print(f"[DEBUG-ESMC-POOLED] seq_feature_2 shape: {seq_feature_2.shape}")
                            
                            features_1.append(seq_feature_1.to(device=device, dtype=model_dtype))
                            features_2.append(seq_feature_2.to(device=device, dtype=model_dtype))
                        else:
                            # Use actual ESMC hidden size if available, otherwise use detected size
                            actual_hidden_size = esmc_hidden_size if esmc_hidden_size is not None else 960
                            feature_1 = torch.zeros(actual_hidden_size, device=device, dtype=model_dtype)
                            feature_2 = torch.zeros(actual_hidden_size, device=device, dtype=model_dtype)
                            features_1.append(feature_1)
                            features_2.append(feature_2)
                    except Exception as e:
                        # Use actual ESMC hidden size if available, otherwise use detected size
                        actual_hidden_size = esmc_hidden_size if esmc_hidden_size is not None else 960
                        feature_1 = torch.zeros(actual_hidden_size, device=device, dtype=model_dtype)
                        feature_2 = torch.zeros(actual_hidden_size, device=device, dtype=model_dtype)
                        features_1.append(feature_1)
                        features_2.append(feature_2)
            else:
                # Process sequences using ESM3
                from esm.sdk.api import ESMProtein
                
                for i, (seq_1, seq_2) in enumerate(zip(sequences_1, sequences_2)):
                    try:
                        # Encode first sequence
                        protein_1 = ESMProtein(sequence=seq_1)
                        with torch.no_grad():
                            encoded_protein_1 = self.model.encode(protein_1)
                        
                        # Encode second sequence
                        protein_2 = ESMProtein(sequence=seq_2)
                        with torch.no_grad():
                            encoded_protein_2 = self.model.encode(protein_2)
                        
                        # Extract sequence tokens
                        if hasattr(encoded_protein_1, 'sequence') and hasattr(encoded_protein_2, 'sequence'):
                            seq_tokens_1 = getattr(encoded_protein_1, 'sequence')
                            seq_tokens_2 = getattr(encoded_protein_2, 'sequence')
                            
                            if torch.is_tensor(seq_tokens_1) and torch.is_tensor(seq_tokens_2):
                                # Convert tokens to embedding dimension
                                seq_feature_1 = seq_tokens_1.float().unsqueeze(-1).expand(-1, hidden_size)
                                seq_feature_2 = seq_tokens_2.float().unsqueeze(-1).expand(-1, hidden_size)
                                
                                # Truncate or pad to fixed length
                                if len(seq_feature_1) > self.fixed_seq_length:
                                    seq_feature_1 = seq_feature_1[:self.fixed_seq_length, :]
                                elif len(seq_feature_1) < self.fixed_seq_length:
                                    padding_size = self.fixed_seq_length - len(seq_feature_1)
                                    padding = torch.zeros(padding_size, hidden_size, device=device, dtype=model_dtype)
                                    seq_feature_1 = torch.cat([seq_feature_1, padding])
                                
                                if len(seq_feature_2) > self.fixed_seq_length:
                                    seq_feature_2 = seq_feature_2[:self.fixed_seq_length, :]
                                elif len(seq_feature_2) < self.fixed_seq_length:
                                    padding_size = self.fixed_seq_length - len(seq_feature_2)
                                    padding = torch.zeros(padding_size, hidden_size, device=device, dtype=model_dtype)
                                    seq_feature_2 = torch.cat([seq_feature_2, padding])
                                
                                # Mean pooling to get sequence representation
                                seq_feature_1 = seq_feature_1.mean(dim=0)  # [hidden_size]
                                seq_feature_2 = seq_feature_2.mean(dim=0)  # [hidden_size]
                                
                                features_1.append(seq_feature_1.to(device=device, dtype=model_dtype))
                                features_2.append(seq_feature_2.to(device=device, dtype=model_dtype))
                            else:
                                # Create zero vectors
                                feature_1 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                                feature_2 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                                features_1.append(feature_1)
                                features_2.append(feature_2)
                        else:
                            # Create zero vectors
                            feature_1 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                            feature_2 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                            features_1.append(feature_1)
                            features_2.append(feature_2)
                    except Exception as e:
                        feature_1 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                        feature_2 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                        features_1.append(feature_1)
                        features_2.append(feature_2)
            
            if features_1 and features_2:
                stacked_features_1 = torch.stack(features_1)  # [batch_size, hidden_size]
                stacked_features_2 = torch.stack(features_2)  # [batch_size, hidden_size]
                stacked_features = torch.cat([stacked_features_1, stacked_features_2], dim=1)  # [batch_size, hidden_size*2]
                # print(f"[DEBUG-SEQUENCES] Created stacked_features: {stacked_features.shape}")
            else:
                batch_size = 1
                stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
                # print(f"[DEBUG-SEQUENCES-EMPTY] Created zero stacked_features: {stacked_features.shape}")
        
        else:
            # print(f"[DEBUG-FALLBACK] Using fallback branch - no valid input found")
            batch_size = 1
            stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
            # print(f"[DEBUG-FALLBACK] Created zero stacked_features: {stacked_features.shape}")
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # Debug: print shape before classification head (commented out for production)
        # if not hasattr(self, '_forward_debug_count'):
        #     self._forward_debug_count = 0
        # if self._forward_debug_count < 5:
        #     print(f"\n[DEBUG] Forward pass {self._forward_debug_count + 1}:")
        #     print(f"  stacked_features.shape: {stacked_features.shape}")
        #     print(f"  Expected shape: [batch_size, {hidden_size * 2}]")
        #     print(f"  classification_head first layer input size: {self.classification_head[0].in_features}")
        #     if "tokens" in inputs_1 and "tokens" in inputs_2:
        #         print(f"  Input source: tokens")
        #     elif "embeddings" in inputs_1 and "embeddings" in inputs_2:
        #         print(f"  Input source: embeddings")
        #     elif "sequences" in inputs_1 and "sequences" in inputs_2:
        #         print(f"  Input source: sequences")
        #     else:
        #         print(f"  Input source: unknown/fallback")
        #     self._forward_debug_count += 1
        
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
        elif stage == "valid":
            # 收集验证损失用于epoch结束时计算平均值
            self.valid_outputs.append(loss.detach())
        elif stage == "test":
            # 收集测试损失用于epoch结束时计算平均值
            self.test_outputs.append(loss.detach())

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
        if len(self.test_outputs) > 0:
            log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))
        else:
            log_dict["test_loss"] = torch.tensor(0.0)

        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        if len(self.valid_outputs) > 0:
            log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))
        else:
            log_dict["valid_loss"] = torch.tensor(0.0)

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)

    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        重写保存方法，保存分类头权重和LoRA权重（如果使用了LoRA）
        """
        import os
        import torch
        
        try:
            # 创建保存目录
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 创建保存的状态字典
            state_dict = {} if save_info is None else save_info.copy()
            state_dict["num_labels"] = self.num_labels
            state_dict["fixed_seq_length"] = self.fixed_seq_length
            state_dict["task"] = "pair_classification"
            
            total_params = 0
            
            # 保存分类头的权重
            if hasattr(self, 'classification_head') and self.classification_head is not None:
                classification_head_state = self.classification_head.state_dict()
                state_dict["classification_head"] = classification_head_state
                
                param_count = sum(p.numel() for p in self.classification_head.parameters())
                total_params += param_count
                # print(f"保存分类头权重:")
                # print(f"  - 参数数量: {param_count:,}")
            
            # 检查是否使用了LoRA，如果是则保存LoRA参数
            from saprot.utils.esm3_lora import ESM3LoRAWrapper
            if isinstance(self.model, ESM3LoRAWrapper):
                lora_state = self.model.get_lora_state_dict()
                state_dict["lora"] = lora_state
                state_dict["lora_config"] = {
                    "r": self.model.r,
                    "alpha": self.model.alpha,
                    "dropout": self.model.dropout,
                    "target_modules": self.model.target_modules
                }
                
                lora_param_count = sum(p.numel() for p in lora_state.values())
                total_params += lora_param_count
                # print(f"保存LoRA权重:")
                # print(f"  - LoRA参数数量: {lora_param_count:,}")
                # print(f"  - LoRA rank: {self.model.r}")
                # print(f"  - Target modules: {len(self.model.lora_layers)}")
            
            # print(f"  - 总参数数量: {total_params:,}")
            # print(f"  - 保存路径: {save_path}")
            
            if not save_weights_only:
                # 如果需要保存训练状态
                state_dict["global_step"] = self.step
                state_dict["epoch"] = self.epoch
                state_dict["best_value"] = getattr(self, "best_value", None)
                
                if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
                    state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
                
                if hasattr(self, 'optimizer') and self.optimizer is not None:
                    state_dict["optimizer"] = self.optimizer.state_dict()
            
            # 保存到文件
            torch.save(state_dict, save_path)
            
            # 验证保存的文件大小
            saved_size = os.path.getsize(save_path) / (1024 * 1024)
            # print(f"模型权重保存成功: {saved_size:.2f} MB")
                
        except Exception as e:
            print(f"保存分类头权重失败: {str(e)}")
            # 尝试保存到当前目录作为备份
            try:
                fallback_path = os.path.join(os.getcwd(), 'pair_classification_head_checkpoint.pt')
                if hasattr(self, 'classification_head'):
                    state_dict = {"classification_head": self.classification_head.state_dict()}
                    torch.save(state_dict, fallback_path)
                    print(f"备用保存成功: {fallback_path}")
            except Exception as e2:
                print(f"备用保存也失败: {str(e2)}")
                raise e

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载分类头权重
        """
        import torch
        import os
        
        # 如果是目录，构造完整的文件路径
        if os.path.isdir(checkpoint_path):
            basename = os.path.basename(checkpoint_path)
            checkpoint_file = os.path.join(checkpoint_path, f"{basename}.pt")
            if os.path.exists(checkpoint_file):
                checkpoint_path = checkpoint_file
            else:
                # print(f"在目录 {checkpoint_path} 中未找到权重文件 {basename}.pt")
                return
        
        if not os.path.exists(checkpoint_path):
            print(f"权重文件不存在: {checkpoint_path}")
            return
        
        try:
            # 加载权重
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 验证是否为分类头权重文件
            if "classification_head" in state_dict:
                # 新格式：包含分类头（和可能的LoRA权重）
                classification_head_state = state_dict["classification_head"]
                num_labels = state_dict.get("num_labels", self.num_labels)
                fixed_seq_length = state_dict.get("fixed_seq_length", self.fixed_seq_length)
                
                print(f"加载权重:")
                print(f"  - 文件: {checkpoint_path}")
                print(f"  - 标签数: {num_labels}")
                print(f"  - 序列长度: {fixed_seq_length}")
                
                # 验证维度匹配
                if num_labels == self.num_labels and fixed_seq_length == self.fixed_seq_length:
                    self.classification_head.load_state_dict(classification_head_state)
                    print(f"分类头权重加载成功")
                    
                    # 检查是否有LoRA权重
                    if "lora" in state_dict:
                        from saprot.utils.esm3_lora import ESM3LoRAWrapper
                        if isinstance(self.model, ESM3LoRAWrapper):
                            lora_state = state_dict["lora"]
                            lora_config = state_dict.get("lora_config", {})
                            
                            print(f"加载LoRA权重:")
                            print(f"  - LoRA rank: {lora_config.get('r', 'unknown')}")
                            print(f"  - LoRA参数数量: {sum(p.numel() for p in lora_state.values()):,}")
                            
                            self.model.load_lora_state_dict(lora_state)
                            print(f"LoRA权重加载成功")
                        else:
                            print(f"检查点包含LoRA权重，但当前模型未启用LoRA")
                    
                else:
                    print(f"维度不匹配: 期望({self.fixed_seq_length}, {self.num_labels}), 实际({fixed_seq_length}, {num_labels})")
                    
            elif "model" in state_dict and any("classification_head" in k for k in state_dict["model"].keys()):
                # 旧格式：包含整个模型，提取分类头部分
                model_state = state_dict["model"]
                classification_head_state = {
                    k.replace("classification_head.", ""): v 
                    for k, v in model_state.items() 
                    if k.startswith("classification_head.")
                }
                if classification_head_state:
                    self.classification_head.load_state_dict(classification_head_state)
                    print(f"从完整模型权重中提取并加载分类头")
                else:
                    print(f"在模型权重中未找到分类头参数")
            else:
                print(f"Unrecognized weight file format")
                
        except Exception as e:
            print(f"加载分类头权重失败: {str(e)}")