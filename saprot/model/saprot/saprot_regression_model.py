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
            
            # #region agent log
            # Debug logging for spearman calculation
            try:
                print(f"\n[DEBUG compute_spearman]")
                print(f"  num_samples={len(preds_np)}")
                print(f"  preds_unique_count={len(np.unique(preds_np))}, targets_unique_count={len(np.unique(targets_np))}")
                print(f"  preds: min={np.min(preds_np):.6f}, max={np.max(preds_np):.6f}, std={np.std(preds_np):.6f}")
                print(f"  targets: min={np.min(targets_np):.6f}, max={np.max(targets_np):.6f}, std={np.std(targets_np):.6f}")
                print(f"  preds_first_10={preds_np[:10].tolist() if len(preds_np) >= 10 else preds_np.tolist()}")
                print(f"  targets_first_10={targets_np[:10].tolist() if len(targets_np) >= 10 else targets_np.tolist()}")
            except Exception as e:
                pass
            # #endregion
            
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

from ..model_interface import register_model
from .base import SaprotBaseModel
# 导入学习率调度器 - 修复导入路径
from utils.lr_scheduler import ConstantLRScheduler, CosineAnnealingLRScheduler, Esm2LRScheduler


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    # ESM3的hidden_dim是1536
    ESM3_HIDDEN_DIM = 1536
    
    def __init__(self, test_result_path: str = None, fixed_seq_length: int = 2048, base_model_type: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            fixed_seq_length: 固定序列长度，用于截断或padding（现在主要用于兼容性）
            base_model_type: 'esm3' or 'esmc', explicitly specify model type
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        self.fixed_seq_length = fixed_seq_length
        self.base_model_type = base_model_type  # 保存base_model_type
        super().__init__(task="regression", **kwargs)
        
        # 创建回归头：输入维度是ESM3的hidden_dim (1536)
        # 使用mean pooling后的全局特征作为输入
        self.regression_head = torch.nn.Linear(self.ESM3_HIDDEN_DIM, 1)
        
        print(f"[INFO] 创建回归头: {self.ESM3_HIDDEN_DIM} -> 1 (使用ESM3 hidden_dim)")
        
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
            # [batch_size, seq_len, hidden_dim] 的情况，先平均池化
            features = features.mean(dim=2)  # [batch_size, seq_len]
            return self._pad_or_truncate_features(features, target_length)
        else:
            raise ValueError(f"不支持的特征维度: {features.shape}")
    
    def forward(self, inputs=None, coords=None, sequences=None, embeddings=None, tokens=None, structure_info=None, **kwargs):
        if structure_info:
            # To be implemented
            raise NotImplementedError

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
        
        # #region agent log
        # Debug logging for input analysis
        try:
            print(f"\n[DEBUG forward_input]")
            if "tokens" in inputs:
                t = inputs["tokens"]
                print(f"  tokens_shape={list(t.shape)}, dtype={t.dtype}, device={t.device}")
                print(f"  tokens: min={t.float().min().item():.4f}, max={t.float().max().item():.4f}, mean={t.float().mean().item():.4f}")
                print(f"  tokens_nonzero_ratio={(t != 0).sum().item() / t.numel():.4f}")
            elif "embeddings" in inputs:
                print(f"  has_embeddings=True, shape={list(inputs['embeddings'].shape)}")
            elif "sequences" in inputs:
                print(f"  has_sequences=True, num_sequences={len(inputs['sequences'])}")
            else:
                print(f"  input_keys={list(inputs.keys())}")
        except Exception as e:
            pass
        # #endregion
        
        # 优先处理tokens - 需要通过ESM3获取语义嵌入
        if "tokens" in inputs:
            tokens = inputs["tokens"].to(device=device)
            batch_size = tokens.shape[0]
            
            # 关键修复：使用ESM3模型获取语义嵌入，而不是直接使用token IDs
            try:
                from esm.sdk.api import ESMProteinTensor
                
                features = []
                model_type = getattr(self, 'model_type', 'esm3')
                
                for i in range(batch_size):
                    sample_tokens = tokens[i]  # [seq_len]
                    
                    # 找到非padding的实际序列长度
                    non_zero_mask = sample_tokens != 0
                    actual_len = non_zero_mask.sum().item()
                    if actual_len == 0:
                        actual_len = 1  # 至少保留一个token
                    
                    # 截取实际的tokens（去除padding）
                    actual_tokens = sample_tokens[:actual_len]
                    
                    try:
                        # 确保tokens是long类型（ESM3期望的输入类型）
                        sequence_tokens_input = actual_tokens.unsqueeze(0).long().to(device)
                        
                        # 使用ESM3模型的forward获取嵌入
                        # 注意：需要确保模型在正确的数据类型下运行
                        with torch.set_grad_enabled(self.training):
                            # 使用autocast确保数据类型一致性
                            with torch.cuda.amp.autocast(enabled=True, dtype=model_dtype):
                                output = self.model.forward(
                                    sequence_tokens=sequence_tokens_input,
                                )
                            
                            # 调试：打印output的属性
                            if i == 0 and batch_size > 0:
                                output_attrs = [attr for attr in dir(output) if not attr.startswith('_')]
                                print(f"\n[DEBUG ESM3 output] attributes: {output_attrs}")
                                for attr in ['embeddings', 'sequence_logits', 'logits', 'hidden_states']:
                                    if hasattr(output, attr):
                                        val = getattr(output, attr)
                                        if val is not None:
                                            if hasattr(val, 'shape'):
                                                print(f"  {attr}: shape={val.shape}, dtype={val.dtype}")
                                            else:
                                                print(f"  {attr}: type={type(val)}")
                                        else:
                                            print(f"  {attr}: None")
                            
                            # 从输出中提取嵌入
                            if hasattr(output, 'embeddings') and output.embeddings is not None:
                                # embeddings: [1, seq_len, hidden_dim] 例如 [1, 190, 1536]
                                seq_embedding = output.embeddings.squeeze(0)  # [seq_len, hidden_dim]
                                
                                # 策略：对序列维度做mean pooling，保留hidden_dim作为特征
                                # 这样每个蛋白质得到一个 [hidden_dim] 的向量，携带全局语义信息
                                seq_feature = seq_embedding.mean(dim=0)  # [hidden_dim] = [1536]
                                
                                if i == 0:
                                    print(f"[DEBUG] Using mean pooling over sequence: {seq_embedding.shape} -> {seq_feature.shape}")
                                
                            elif hasattr(output, 'sequence_logits') and output.sequence_logits is not None:
                                # 如果没有embeddings，使用sequence_logits
                                # sequence_logits: [1, seq_len, vocab_size]
                                seq_logits = output.sequence_logits.squeeze(0)  # [seq_len, vocab_size]
                                seq_feature = seq_logits.mean(dim=0)  # [vocab_size]
                            else:
                                # 回退：使用token值本身（但这不理想）
                                print(f"[WARNING] ESM3 output has no embeddings, falling back to token values")
                                seq_feature = actual_tokens.float().to(dtype=model_dtype)
                        
                        # 特征已经是固定维度（hidden_dim=1536），不需要padding
                        seq_feature = seq_feature.to(dtype=model_dtype)
                        features.append(seq_feature)
                        
                    except Exception as e:
                        print(f"[WARNING] ESM3 embedding extraction failed for sample {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        # 回退：创建零向量，维度是ESM3_HIDDEN_DIM
                        features.append(torch.zeros(self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype))
                
                stacked_features = torch.stack(features)  # [batch_size, ESM3_HIDDEN_DIM]
                
            except Exception as e:
                print(f"[WARNING] Token processing failed: {str(e)}, falling back to zero features")
                import traceback
                traceback.print_exc()
                stacked_features = torch.zeros(batch_size, self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype)
        
        # 处理预编码的嵌入
        elif "embeddings" in inputs:
            embeddings = inputs["embeddings"].to(device=device, dtype=model_dtype)
            # 如果是3D嵌入 [batch_size, seq_len, hidden_dim]，做mean pooling得到 [batch_size, hidden_dim]
            if embeddings.dim() == 3:
                stacked_features = embeddings.mean(dim=1)  # [batch_size, hidden_dim]
            else:
                stacked_features = embeddings
        
        elif "sequences" in inputs:
            sequences = inputs["sequences"]
            model_type = getattr(self, 'model_type', 'esm3')
            
            features = []
            for i, seq in enumerate(sequences):
                try:
                    from esm.sdk.api import ESMProtein
                    protein = ESMProtein(sequence=seq)
                    
                    # 使用ESM3 forward获取嵌入
                    with torch.set_grad_enabled(self.training):
                        with torch.cuda.amp.autocast(enabled=True, dtype=model_dtype):
                            # 先encode获取token
                            encoded = self.model.encode(protein)
                            # 再forward获取嵌入
                            output = self.model.forward(sequence_tokens=encoded.sequence.unsqueeze(0).to(device))
                            
                            if hasattr(output, 'embeddings') and output.embeddings is not None:
                                # [1, seq_len, hidden_dim] -> [hidden_dim]
                                seq_feature = output.embeddings.squeeze(0).mean(dim=0)
                            else:
                                seq_feature = torch.zeros(self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype)
                    
                    features.append(seq_feature.to(dtype=model_dtype))
                except Exception as e:
                    print(f"[WARNING] Sequence {i} encoding error: {str(e)}")
                    features.append(torch.zeros(self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype))
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype)

        # 保留原有的ESM和ProtBERT逻辑作为兜底
        elif "inputs" in inputs:
            model_inputs = inputs["inputs"]
            
            # For ESM models
            if hasattr(self.model, "esm"):
                # If backbone is frozen, the embedding will be the average of all residues, else it will be the
                # embedding of the <cls> token.
                if self.freeze_backbone:
                    repr = torch.stack(self.get_hidden_states_from_dict(model_inputs, reduction="mean"))
                    x = self.model.classifier.dropout(repr)
                    x = self.model.classifier.dense(x)
                    x = torch.tanh(x)
                    x = self.model.classifier.dropout(x)
                    logits = self.model.classifier.out_proj(x)
                else:
                   logits = self.model(**model_inputs).logits
                   
                return logits
        
             # For ProtBERT
            elif hasattr(self.model, "bert"):
                # 检查输入的token IDs是否在有效范围内
                vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
                input_ids = model_inputs["input_ids"]
                if torch.max(input_ids) >= vocab_size:
                    # 将超出范围的ID替换为UNK token ID
                    unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                    model_inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
                repr = self.model.bert(**model_inputs).last_hidden_state[:, 0]
                logits = self.model.classifier(repr)
                
                return logits
        
        else:
            # print(f"[回归模型调试] 输入中没有找到tokens、embeddings、sequences或inputs")
            stacked_features = torch.zeros(1, self.ESM3_HIDDEN_DIM, device=device, dtype=model_dtype)
        
        # Ensure stacked_features is on the correct device and dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # print(f"[回归模型调试] 最终特征维度: {stacked_features.shape} (固定长度: {self.fixed_seq_length})")

        # 确保回归头在正确的设备和数据类型上
        self.regression_head = self.regression_head.to(device=device, dtype=model_dtype)
        
        # #region agent log
        # Debug logging for feature and regression head analysis
        try:
            print(f"\n[DEBUG forward_features]")
            print(f"  stacked_features: shape={list(stacked_features.shape)}, dtype={stacked_features.dtype}, device={stacked_features.device}")
            print(f"  stacked_features: min={stacked_features.min().item():.4f}, max={stacked_features.max().item():.4f}, mean={stacked_features.mean().item():.4f}, std={stacked_features.std().item():.4f}")
            print(f"  stacked_features_nonzero_ratio={(stacked_features != 0).sum().item() / stacked_features.numel():.4f}")
            print(f"  regression_head_weight: min={self.regression_head.weight.min().item():.6f}, max={self.regression_head.weight.max().item():.6f}, mean={self.regression_head.weight.mean().item():.6f}")
            if self.regression_head.bias is not None:
                print(f"  regression_head_bias={self.regression_head.bias.item():.6f}")
        except Exception as e:
            pass
        # #endregion
        
        # Forward pass - 不使用squeeze，保持与classification一致
        logits = self.regression_head(stacked_features)
        # print(f"[回归模型调试] 回归输出形状: {logits.shape}")
        
        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        
        # 确保形状匹配：flatten输出和标签
        outputs_flat = outputs.flatten()
        fitness_flat = fitness.flatten()
        
        # #region agent log
        # Debug logging for hypothesis verification
        try:
            print(f"\n[DEBUG loss_func] stage={stage}")
            print(f"  outputs_shape={list(outputs.shape)}, outputs_flat_shape={list(outputs_flat.shape)}")
            print(f"  outputs: min={outputs_flat.min().item():.6f}, max={outputs_flat.max().item():.6f}, mean={outputs_flat.mean().item():.6f}, std={outputs_flat.std().item() if len(outputs_flat) > 1 else 0.0:.6f}")
            print(f"  fitness: min={fitness_flat.min().item():.6f}, max={fitness_flat.max().item():.6f}, mean={fitness_flat.mean().item():.6f}, std={fitness_flat.std().item() if len(fitness_flat) > 1 else 0.0:.6f}")
            print(f"  outputs_first_5={[float(x) for x in outputs_flat[:5].tolist()]}")
            print(f"  fitness_first_5={[float(x) for x in fitness_flat[:5].tolist()]}")
        except Exception as e:
            pass
        # #endregion
        
        loss = torch.nn.functional.mse_loss(outputs_flat, fitness_flat)
        
        # Update metrics - 使用自定义的SimpleRegressionMetrics
        with torch.no_grad():
            for metric_name, metric in self.metrics[stage].items():
                metric.update(outputs_flat.detach(), fitness_flat)
            
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
        
        # #region agent log
        # Debug logging for optimizer initialization
        try:
            print(f"\n[DEBUG init_optimizers]")
            print(f"  esm3_param_count={esm3_param_count}")
            print(f"  regression_head_param_count={regression_head_param_count}")
            print(f"  total_param_count={len(all_params)}")
            print(f"  weight_decay={weight_decay}")
            print(f"  init_lr={self.lr_scheduler_kwargs.get('init_lr', 'N/A')}")
            print(f"  lr_scheduler_class={self.lr_scheduler_kwargs.get('class', 'N/A')}")
        except Exception as e:
            pass
        # #endregion

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
        elif lr_scheduler_name == "CosineAnnealingLRScheduler":
            lr_scheduler_cls = CosineAnnealingLRScheduler
        elif lr_scheduler_name == "Esm2LRScheduler":
            lr_scheduler_cls = Esm2LRScheduler
        elif hasattr(torch.optim.lr_scheduler, lr_scheduler_name):
            # 如果是PyTorch内置的调度器
            lr_scheduler_cls = getattr(torch.optim.lr_scheduler, lr_scheduler_name)
        else:
            # print(f" 未知的学习率调度器: {lr_scheduler_name}, 使用ConstantLRScheduler")
            lr_scheduler_cls = ConstantLRScheduler
            
        self.lr_scheduler = lr_scheduler_cls(self.optimizer, **tmp_kwargs)
        
        # print(f"优化器重新初始化完成，总参数组数: {len(optimizer_grouped_parameters)}")
        # print(f"学习率调度器: {lr_scheduler_name}")
        # print(f"初始学习率: {self.lr_scheduler_kwargs.get('init_lr', 'N/A')}")

    # training_step和on_before_optimizer_step方法已移除
    # 这些功能现在由纯PyTorch训练循环处理

    def on_test_epoch_end(self):
        # 打印回归头权重信息
        # self._print_regression_head_weights("测试")
        
        log_dict = self.get_log_dict("test")
        
        if len(self.test_outputs) > 0:
            log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))
        else:
            log_dict["test_loss"] = torch.tensor(0.0)

        # 如果需要保存测试结果到文件
        if self.test_result_path is not None:
            metrics_obj = self.metrics["test"].get("test_metrics")
            if metrics_obj and len(metrics_obj.preds) > 0:
                preds = torch.cat(metrics_obj.preds)
                targets = torch.cat(metrics_obj.targets)
                
                if dist.is_initialized() and dist.get_rank() == 0:
                    with open(self.test_result_path, 'w') as w:
                        w.write("pred\ttarget\n")
                        for pred, target in zip(preds, targets):
                            pred_arr = pred.flatten().tolist()
                            target_arr = target.flatten().tolist()
                            pred_str = str(pred_arr[0]) if len(pred_arr) == 1 else ','.join(map(str, pred_arr))
                            target_str = str(target_arr[0]) if len(target_arr) == 1 else ','.join(map(str, target_arr))
                            w.write(f"{pred_str}\t{target_str}\n")
                elif not dist.is_initialized():
                    with open(self.test_result_path, 'w') as w:
                        w.write("pred\ttarget\n")
                        for pred, target in zip(preds, targets):
                            pred_arr = pred.flatten().tolist()
                            target_arr = target.flatten().tolist()
                            pred_str = str(pred_arr[0]) if len(pred_arr) == 1 else ','.join(map(str, pred_arr))
                            target_str = str(target_arr[0]) if len(target_arr) == 1 else ','.join(map(str, target_arr))
                            w.write(f"{pred_str}\t{target_str}\n")

        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

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
            state_dict["task"] = "regression"
            
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
                fallback_path = os.path.join(os.getcwd(), 'regression_head_checkpoint.pt')
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
            # Load weights
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Verify if this is a regression head weight file
            if "regression_head" in state_dict:
                # New format: contains regression head (and possibly LoRA weights)
                regression_head_state = state_dict["regression_head"]
                fixed_seq_length = state_dict.get("fixed_seq_length", self.fixed_seq_length)
                
                # print(f"Loading weights:")
                # print(f"  - File: {checkpoint_path}")
                # print(f"  - Sequence length: {fixed_seq_length}")
                
                # Verify dimension matching
                if fixed_seq_length == self.fixed_seq_length:
                    self.regression_head.load_state_dict(regression_head_state)
                    # print(f"Regression head weights loaded successfully")
                    
                    # Check if there are LoRA weights
                    if "lora" in state_dict:
                        from saprot.utils.esm3_lora import ESM3LoRAWrapper
                        if isinstance(self.model, ESM3LoRAWrapper):
                            lora_state = state_dict["lora"]
                            lora_config = state_dict.get("lora_config", {})
                            
                            # print(f"Loading LoRA weights:")
                            # print(f"  - LoRA rank: {lora_config.get('r', 'unknown')}")
                            # print(f"  - LoRA parameters: {sum(p.numel() for p in lora_state.values()):,}")
                            
                            self.model.load_lora_state_dict(lora_state)
                            # print(f"LoRA weights loaded successfully")
                        else:
                            print(f"Checkpoint contains LoRA weights, but current model does not have LoRA enabled")
                    
                else:
                    print(f"Dimension mismatch: expected({self.fixed_seq_length}, 1), got({fixed_seq_length}, 1)")
                    
            elif "model" in state_dict and any("regression_head" in k for k in state_dict["model"].keys()):
                # Old format: contains full model, extract regression head
                model_state = state_dict["model"]
                regression_head_state = {
                    k.replace("regression_head.", ""): v 
                    for k, v in model_state.items() 
                    if k.startswith("regression_head.")
                }
                if regression_head_state:
                    self.regression_head.load_state_dict(regression_head_state)
                    # print(f"Extracted and loaded regression head from full model weights")
                else:
                    print(f"Regression head parameters not found in model weights")
            else:
                print(f"Unrecognized weight file format")
                
        except Exception as e:
            print(f"Failed to load regression head weights: {str(e)}")

    def on_validation_epoch_end(self):
        # 打印回归头权重信息
        # self._print_regression_head_weights("验证")
        
        log_dict = self.get_log_dict("valid")
        if len(self.valid_outputs) > 0:
            log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))
        else:
            log_dict["valid_loss"] = torch.tensor(0.0)

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        
        self.plot_valid_metrics_curve(log_dict)

    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        super().on_train_epoch_end()  # 调用父类方法
        # 打印回归头权重信息
        # self._print_regression_head_weights("训练")

    def _print_regression_head_weights(self, stage_name):
        """打印回归头权重统计信息"""
        pass

    def _check_optimizer_state(self):
        """检查优化器状态以诊断训练问题"""
        if hasattr(self, 'optimizer'):
            # print("\n=== 优化器状态诊断 ===")
            
            # 检查学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            # print(f"当前学习率: {current_lr}")
            
            if current_lr == 0:
                # print("学习率为0，这会阻止参数更新!")
                pass
            elif current_lr < 1e-8:
                # print(" 学习率非常小，可能导致缓慢的收敛")
                pass
            
            # 检查回归头参数是否在优化器中
            regression_head_param_ids = {id(p) for p in self.regression_head.parameters()}
            optimizer_param_ids = set()
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    optimizer_param_ids.add(id(param))
            
            missing_params = regression_head_param_ids - optimizer_param_ids
            if missing_params:
                # print("回归头参数不在优化器中!")
                pass
            else:
                # print("回归头参数已在优化器中")
                pass
            
            # 检查梯度
            total_grad_norm = 0.0
            param_count = 0
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
            
            if param_count > 0:
                total_grad_norm = total_grad_norm ** 0.5
                # print(f"总梯度范数: {total_grad_norm:.6f}")
                # print(f"有梯度的参数数: {param_count}")
            else:
                # print("没有参数有梯度!")
                pass
            
            # print("=" * 30)
        pass

    def _verify_regression_head_in_optimizer(self):
        """验证回归头参数是否包含在优化器中"""
        pass