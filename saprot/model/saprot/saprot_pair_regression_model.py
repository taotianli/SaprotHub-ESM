import os

# 禁用transformers的accelerate集成，避免numpy 2.x兼容性问题
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'

import torch
import torch.distributed as dist

# 自定义回归指标实现，避免导入torchmetrics（会触发accelerate/numpy兼容性问题）
class SimpleRegressionMetrics:
    """简单的回归指标计算，替代torchmetrics"""
    def __init__(self):
        self.reset()
    
    def update(self, preds, target):
        """更新统计"""
        if not torch.is_tensor(preds):
            preds = torch.tensor(preds)
        if not torch.is_tensor(target):
            target = torch.tensor(target)
        
        # 转换为float并展平
        preds = preds.float().flatten()
        target = target.float().flatten()
        
        self.preds.append(preds.detach().cpu())
        self.targets.append(target.detach().cpu())
    
    def compute_mse(self):
        """计算均方误差"""
        if len(self.preds) == 0:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        return torch.mean((preds - targets) ** 2).item()
    
    def compute_rmse(self):
        """计算均方根误差 (RMSE = sqrt(MSE))"""
        mse = self.compute_mse()
        return mse ** 0.5
    
    def compute_pearson(self):
        """计算皮尔逊相关系数"""
        if len(self.preds) == 0:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        # 计算皮尔逊相关系数
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)
        
        # 计算标准差
        std_x = torch.sqrt(torch.sum(vx ** 2))
        std_y = torch.sqrt(torch.sum(vy ** 2))
        
        # 避免除以零
        if std_x == 0 or std_y == 0:
            return 0.0
        
        corr = torch.sum(vx * vy) / (std_x * std_y)
        result = corr.item()
        
        # 检查是否为nan
        if result != result:  # NaN check
            return 0.0
        
        return result
    
    def compute_spearman(self):
        """计算斯皮尔曼相关系数（使用排名）"""
        if len(self.preds) == 0:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        # Spearman需要至少3个样本才能有效计算
        if len(preds) < 3:
            return self.compute_pearson()
        
        # 使用scipy计算，通过np.array()显式转换避免NumPy 2.x兼容性问题
        try:
            from scipy.stats import spearmanr
            import numpy as np
            import warnings
            
            # 显式转换为numpy array，避免_CopyMode.IF_NEEDED错误
            preds_np = np.array(preds.cpu().numpy(), dtype=np.float64)
            targets_np = np.array(targets.cpu().numpy(), dtype=np.float64)
            
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                corr, _ = spearmanr(preds_np, targets_np)
            
            # 检查是否为nan
            if np.isnan(corr):
                return 0.0
            return float(corr)
        except Exception as e:
            print(f"Spearman calculation failed: {e}")
            return 0.0
    
    def compute_r2(self):
        """计算R²分数"""
        if len(self.preds) == 0:
            return 0.0
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        # R² = 1 - SS_res / SS_tot
        ss_res = torch.sum((targets - preds) ** 2)
        ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        return (1 - ss_res / ss_tot).item()
    
    def reset(self):
        """重置统计"""
        self.preds = []
        self.targets = []

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel
# 导入学习率调度器 - 修复导入路径
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotPairRegressionModel(SaprotBaseModel):
    def __init__(self, fixed_seq_length: int = 2048, base_model_type: str = None, optimizer_kwargs=None, lr_scheduler_kwargs=None, **kwargs):
        """
        Args:
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
        
        self.fixed_seq_length = fixed_seq_length
        super().__init__(task="base", **kwargs)
        
        # 回归头将在initialize_model中创建
        # print(f"回归头将在initialize_model中创建")
        
        # 重新初始化优化器以包含回归头参数
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
        
        # 默认返回 ESM3-open 的隐藏维度
        return 2560

    def initialize_model(self):
        """初始化ESM3模型和回归头"""
        super().initialize_model()
        
        # 动态检测模型的隐藏维度
        hidden_size = self._get_model_hidden_size()
        model_type = type(self.model).__name__
        # print(f"[DEBUG] Pair Regressor detected hidden_size: {hidden_size}, model_type: {model_type}")
        
        # 对于pair回归，我们需要两倍的hidden_size，因为要处理两个序列
        hidden_size = hidden_size * 2
        self.pair_hidden_size = hidden_size  # 保存用于forward中的标准化
        
        # 创建简单的回归头：单一线性层
        # 使用普通标准化（在forward中计算），不添加额外的LayerNorm层
        self.regression_head = torch.nn.Linear(hidden_size, 1)
        
        # 确保回归头参数可训练
        for param in self.regression_head.parameters():
            param.requires_grad = True
        
        # print(f"Pair regressor created with hidden_size={hidden_size} (single={hidden_size//2}) for {model_type}")
        
        # 重新初始化优化器以包含回归头参数
        self.init_optimizers()

    def initialize_metrics(self, stage):
        # 使用自定义的SimpleRegressionMetrics，避免torchmetrics的依赖问题
        return {f"{stage}_metrics": SimpleRegressionMetrics()}

    def get_log_dict(self, stage):
        """从metrics中提取日志字典"""
        log_dict = {}
        metrics_obj = self.metrics[stage].get(f"{stage}_metrics")
        if metrics_obj:
            # 使用RMSE而不是MSE，因为图表标签显示的是RMSE
            log_dict[f"{stage}_loss"] = metrics_obj.compute_rmse()
            log_dict[f"{stage}_spearman"] = metrics_obj.compute_spearman()
            log_dict[f"{stage}_R2"] = metrics_obj.compute_r2()
            log_dict[f"{stage}_pearson"] = metrics_obj.compute_pearson()
        return log_dict

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
        
        # 使用统一的方法检测模型隐藏维度
        hidden_size = self._get_model_hidden_size()
        
        # 优先处理tokens - 使用ESM3获取真正的语义嵌入
        if "tokens" in inputs_1 and "tokens" in inputs_2:
            tokens_1 = inputs_1["tokens"].to(device=device)
            tokens_2 = inputs_2["tokens"].to(device=device)
            batch_size = tokens_1.shape[0]
            
            try:
                features_1 = []
                features_2 = []
                
                for i in range(batch_size):
                    # 处理第一个序列
                    sample_tokens_1 = tokens_1[i]
                    non_zero_mask_1 = sample_tokens_1 != 0
                    actual_len_1 = max(non_zero_mask_1.sum().item(), 1)
                    actual_tokens_1 = sample_tokens_1[:actual_len_1]
                    
                    # 处理第二个序列
                    sample_tokens_2 = tokens_2[i]
                    non_zero_mask_2 = sample_tokens_2 != 0
                    actual_len_2 = max(non_zero_mask_2.sum().item(), 1)
                    actual_tokens_2 = sample_tokens_2[:actual_len_2]
                    
                    try:
                        # 使用ESM3获取第一个序列的嵌入
                        with torch.set_grad_enabled(self.training):
                            with torch.cuda.amp.autocast(enabled=True, dtype=model_dtype):
                                output_1 = self.model.forward(
                                    sequence_tokens=actual_tokens_1.unsqueeze(0).long().to(device)
                                )
                                output_2 = self.model.forward(
                                    sequence_tokens=actual_tokens_2.unsqueeze(0).long().to(device)
                                )
                            
                            # 提取嵌入并做mean pooling
                            if hasattr(output_1, 'embeddings') and output_1.embeddings is not None:
                                seq_feature_1 = output_1.embeddings.squeeze(0).mean(dim=0)
                            else:
                                seq_feature_1 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                            
                            if hasattr(output_2, 'embeddings') and output_2.embeddings is not None:
                                seq_feature_2 = output_2.embeddings.squeeze(0).mean(dim=0)
                            else:
                                seq_feature_2 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                        
                        features_1.append(seq_feature_1.to(dtype=model_dtype))
                        features_2.append(seq_feature_2.to(dtype=model_dtype))
                        
                    except Exception as e:
                        features_1.append(torch.zeros(hidden_size, device=device, dtype=model_dtype))
                        features_2.append(torch.zeros(hidden_size, device=device, dtype=model_dtype))
                
                # 堆叠特征并连接
                stacked_1 = torch.stack(features_1)  # [batch_size, hidden_size]
                stacked_2 = torch.stack(features_2)  # [batch_size, hidden_size]
                stacked_features = torch.cat([stacked_1, stacked_2], dim=1)  # [batch_size, hidden_size*2]
                
            except Exception as e:
                batch_size = tokens_1.shape[0] if tokens_1.dim() > 0 else 1
                stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs_1 and "embeddings" in inputs_2:
            embeddings_1 = inputs_1["embeddings"].to(device=device, dtype=model_dtype)
            embeddings_2 = inputs_2["embeddings"].to(device=device, dtype=model_dtype)
            
            # 如果是高维嵌入，需要转换为固定长度
            if embeddings_1.dim() == 3:
                embeddings_1 = embeddings_1.mean(dim=1)  # [batch_size, hidden_size]
            if embeddings_2.dim() == 3:
                embeddings_2 = embeddings_2.mean(dim=1)  # [batch_size, hidden_size]
                
            stacked_features = torch.cat([embeddings_1, embeddings_2], dim=1)  # [batch_size, hidden_size*2]
        
        elif "sequences" in inputs_1 and "sequences" in inputs_2:
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
                            # embeddings shape: [batch_size, seq_len, hidden_dim] or [seq_len, hidden_dim]
                            # Get actual hidden size from embeddings
                            if esmc_hidden_size is None:
                                esmc_hidden_size = logits_output_1.embeddings.shape[-1]
                            
                            # 处理可能的batch维度
                            emb_1 = logits_output_1.embeddings
                            emb_2 = logits_output_2.embeddings
                            
                            # 如果是3D (batch_size, seq_len, hidden)，先去掉batch维度或对序列维度取平均
                            if emb_1.dim() == 3:
                                # 对序列维度(dim=1)取平均，然后squeeze掉batch维度
                                seq_feature_1 = emb_1.mean(dim=1).squeeze(0).float()  # [esmc_hidden_size]
                            else:
                                # 如果是2D (seq_len, hidden)，对序列维度(dim=0)取平均
                                seq_feature_1 = emb_1.mean(dim=0).float()  # [esmc_hidden_size]
                            
                            if emb_2.dim() == 3:
                                seq_feature_2 = emb_2.mean(dim=1).squeeze(0).float()  # [esmc_hidden_size]
                            else:
                                seq_feature_2 = emb_2.mean(dim=0).float()  # [esmc_hidden_size]
                            
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
                                # 将tokens转换为嵌入维度
                                seq_feature_1 = seq_tokens_1.float().unsqueeze(-1).expand(-1, hidden_size)
                                seq_feature_2 = seq_tokens_2.float().unsqueeze(-1).expand(-1, hidden_size)
                                
                                # 截断或padding到固定长度
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
                                
                                # 平均池化得到序列表示
                                seq_feature_1 = seq_feature_1.mean(dim=0)  # [hidden_size]
                                seq_feature_2 = seq_feature_2.mean(dim=0)  # [hidden_size]
                                
                                features_1.append(seq_feature_1.to(device=device, dtype=model_dtype))
                                features_2.append(seq_feature_2.to(device=device, dtype=model_dtype))
                            else:
                                # 创建零向量
                                feature_1 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                                feature_2 = torch.zeros(hidden_size, device=device, dtype=model_dtype)
                                features_1.append(feature_1)
                                features_2.append(feature_2)
                        else:
                            # 创建零向量
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
            else:
                batch_size = 1
                stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
        
        # 保留原有的ESM和ProtBERT逻辑作为兜底
        elif "inputs" in inputs_1 and "inputs" in inputs_2:
            model_inputs_1 = inputs_1["inputs"]
            model_inputs_2 = inputs_2["inputs"]
            
            if self.freeze_backbone:
                hidden_1 = torch.stack(self.get_hidden_states_from_dict(model_inputs_1, reduction="mean"))
                hidden_2 = torch.stack(self.get_hidden_states_from_dict(model_inputs_2, reduction="mean"))
            else:
                # If "esm" is not in the model, use "bert" as the backbone
                backbone = self.model.esm if hasattr(self.model, "esm") else self.model.bert
                hidden_1 = backbone(**model_inputs_1)[0][:, 0, :]
                hidden_2 = backbone(**model_inputs_2)[0][:, 0, :]
            
            stacked_features = torch.cat([hidden_1, hidden_2], dim=-1)
        
        else:
            batch_size = 1
            stacked_features = torch.zeros(batch_size, hidden_size * 2, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # 使用普通标准化：(x - mean) / (std + eps)
        eps = 1e-6
        feat_mean = stacked_features.mean(dim=-1, keepdim=True)
        feat_std = stacked_features.std(dim=-1, keepdim=True)
        normalized_features = (stacked_features - feat_mean) / (feat_std + eps)
        
        # 确保回归头在正确的设备和数据类型上
        self.regression_head = self.regression_head.to(device=device, dtype=model_dtype)
        
        # Forward pass - 不使用squeeze，保持与classification一致
        logits = self.regression_head(normalized_features)
        
        return logits

    def loss_func(self, stage, logits, labels):
        fitness = labels['labels'].to(logits)
        
        # 确保形状匹配：flatten输出和标签
        logits_flat = logits.flatten()
        fitness_flat = fitness.flatten()
        
        loss = torch.nn.functional.mse_loss(logits_flat, fitness_flat)

        # Update metrics - 使用自定义的SimpleRegressionMetrics
        with torch.no_grad():
            for metric in self.metrics[stage].values():
                metric.update(logits_flat.detach(), fitness_flat)

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
        """重写优化器初始化，确保包含回归头参数"""
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
        
        # 添加回归头参数
        regression_head_param_count = 0
        if hasattr(self, 'regression_head') and self.regression_head is not None:
            for name, param in self.regression_head.named_parameters():
                if param.requires_grad:
                    full_name = f"regression_head.{name}"
                    all_params.append((full_name, param))
                    regression_head_param_count += 1
                    # print(f"  添加到优化器: {full_name}")

        # print(f"回归头可训练参数数量: {regression_head_param_count}")
        # print(f"总可训练参数数量: {len(all_params)}")

        if not all_params:
            # print("警告: 没有找到需要优化的参数!")
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
            # print(f" 未知的学习率调度器: {lr_scheduler_name}, 使用ConstantLRScheduler")
            lr_scheduler_cls = ConstantLRScheduler
            # 默认使用ConstantLRScheduler，同样过滤参数
            allowed_keys = {'init_lr', 'last_epoch', 'verbose'}
            tmp_kwargs = {k: v for k, v in tmp_kwargs.items() if k in allowed_keys}
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)

    def training_step(self, batch, batch_idx):
        """重写训练步骤，添加详细的梯度监控"""
        inputs, labels = batch
        
        # 前向传播
        outputs = self(**inputs)
        
        # 计算损失
        loss = self.loss_func('train', outputs, labels)
        
        # print(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
        
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
        self.check_save_condition(log_dict["valid_loss"], mode="min")

        self.plot_valid_metrics_curve(log_dict)

    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        重写保存方法，保存回归头权重和LoRA权重（如果使用了LoRA）
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
            state_dict["fixed_seq_length"] = self.fixed_seq_length
            state_dict["task"] = "pair_regression"
            
            total_params = 0
            
            # 保存回归头的权重
            if hasattr(self, 'regression_head') and self.regression_head is not None:
                regression_head_state = self.regression_head.state_dict()
                state_dict["regression_head"] = regression_head_state
                
                param_count = sum(p.numel() for p in self.regression_head.parameters())
                total_params += param_count
                # print(f"保存回归头权重:")
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
            print(f"Failed to save regression head weights: {str(e)}")
            # Try saving to current directory as backup
            try:
                fallback_path = os.path.join(os.getcwd(), 'pair_regression_head_checkpoint.pt')
                if hasattr(self, 'regression_head'):
                    state_dict = {"regression_head": self.regression_head.state_dict()}
                    torch.save(state_dict, fallback_path)
                    print(f"Backup save successful: {fallback_path}")
            except Exception as e2:
                print(f"Backup save also failed: {str(e2)}")
                raise e

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载回归头权重
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
            print(f"Weight file does not exist: {checkpoint_path}")
            return
        
        try:
            # 加载权重
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 验证是否为回归头权重文件
            if "regression_head" in state_dict:
                # 新格式：包含回归头（和可能的LoRA权重）
                regression_head_state = state_dict["regression_head"]
                fixed_seq_length = state_dict.get("fixed_seq_length", self.fixed_seq_length)
                
                print(f"Loading weights:")
                print(f"  - File: {checkpoint_path}")
                print(f"  - Sequence length: {fixed_seq_length}")
                
                # 验证维度匹配
                if fixed_seq_length == self.fixed_seq_length:
                    self.regression_head.load_state_dict(regression_head_state)
                    print(f"Regression head weights loaded successfully")
                    
                    # 检查是否有LoRA权重
                    if "lora" in state_dict:
                        from saprot.utils.esm3_lora import ESM3LoRAWrapper
                        if isinstance(self.model, ESM3LoRAWrapper):
                            lora_state = state_dict["lora"]
                            lora_config = state_dict.get("lora_config", {})
                            
                            print(f"Loading LoRA weights:")
                            print(f"  - LoRA rank: {lora_config.get('r', 'unknown')}")
                            print(f"  - LoRA parameter count: {sum(p.numel() for p in lora_state.values()):,}")
                            
                            self.model.load_lora_state_dict(lora_state)
                            print(f"LoRA weights loaded successfully")
                        else:
                            print(f"Checkpoint contains LoRA weights, but current model does not have LoRA enabled")
                    
                else:
                    print(f"Dimension mismatch: expected ({self.fixed_seq_length}, 1), actual ({fixed_seq_length}, 1)")
                    
            elif "model" in state_dict and any("regression_head" in k for k in state_dict["model"].keys()):
                # 旧格式：包含整个模型，提取回归头部分
                model_state = state_dict["model"]
                regression_head_state = {
                    k.replace("regression_head.", ""): v 
                    for k, v in model_state.items() 
                    if k.startswith("regression_head.")
                }
                if regression_head_state:
                    self.regression_head.load_state_dict(regression_head_state)
                    print(f"Extracted and loaded regression head from full model weights")
                else:
                    print(f"Regression head parameters not found in model weights")
            else:
                print(f"Unrecognized weight file format")
                
        except Exception as e:
            print(f"Failed to load regression head weights: {str(e)}")