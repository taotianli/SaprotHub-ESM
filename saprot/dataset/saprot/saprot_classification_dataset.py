import torch
import json
import random

from ..data_interface import register_dataset
from transformers import AutoTokenizer, EsmTokenizer
from ..lmdb_dataset import *
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


@register_dataset
class SaprotClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str = None,  # Keep parameter for compatibility but not used
                 use_bias_feature: bool = False,
                 max_length: int = 1024,
                 preset_label: int = None,
                 mask_struc_ratio: float = None,
                 mask_seed: int = 20000812,
                 plddt_threshold: float = None,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer (not used for ESM3, kept for compatibility)
            use_bias_feature: If True, structure information will be used
            max_length: Max length of sequence
            preset_label: If not None, all labels will be set to this value
            mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
            mask_seed: Seed for mask_struc_ratio
            plddt_threshold: If not None, mask structure tokens with pLDDT < threshold
            **kwargs:
        """
        # Force num_workers to 0 to avoid multiprocessing CUDA issues
        if 'dataloader_kwargs' not in kwargs:
            kwargs['dataloader_kwargs'] = {}
        kwargs['dataloader_kwargs']['num_workers'] = 0
        
        super().__init__(**kwargs)
        # Don't initialize ESM3 model here to avoid multiprocessing issues
        # It will be initialized in collate_fn or passed from the model
        self.esm_model = None
        
        self.max_length = max_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self.mask_struc_ratio = mask_struc_ratio
        self.mask_seed = mask_seed
        self.plddt_threshold = plddt_threshold

        self.is_saprot_model = True  # Always true for ESM3

    def set_esm_model(self, esm_model):
        """Set the ESM3 model for encoding. This should be called from the main process."""
        self.esm_model = esm_model

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length-2]
        
        # Convert sequence to string format for ESM3
        if isinstance(seq, list):
            seq = "".join(seq)
        
        # Apply masking if needed (simplified for ESM3)
        if self.mask_struc_ratio is not None:
            # Simple random masking for compatibility
            random.seed(self.mask_seed + index)  # Add index to make it deterministic per sample
            seq_list = list(seq)
            mask_num = int(len(seq_list) * self.mask_struc_ratio)
            mask_indices = random.sample(range(len(seq_list)), mask_num)
            for idx in mask_indices:
                seq_list[idx] = 'X'  # Use X for masked tokens
            seq = "".join(seq_list)
        
        # 在主线程中进行ESM3编码
        try:
            if self.esm_model is not None:
                # 使用ESM3模型编码sequence
                print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                
                protein = ESMProtein(sequence=seq)
                with torch.no_grad():  # 编码时不需要梯度
                    encoded_protein = self.esm_model.encode(protein)
                
                # 获取编码结果并打印token信息
                if hasattr(encoded_protein, 'sequence'):
                    seq_repr = encoded_protein.sequence
                    if torch.is_tensor(seq_repr):
                        # 应用平均池化获得固定长度的representation
                        if seq_repr.dim() > 1:
                            sequence_embedding = seq_repr.mean(dim=0)  # [seq_len, hidden_dim] -> [hidden_dim]
                        else:
                            sequence_embedding = seq_repr
                        
                        print(f"[数据集调试] 索引 {index} - ESM3原始输出形状: {seq_repr.shape}, dtype: {seq_repr.dtype}")
                        print(f"[数据集调试] 索引 {index} - 池化后嵌入形状: {sequence_embedding.shape}")
                        print(f"[数据集调试] 索引 {index} - Token统计: min={seq_repr.min().item():.4f}, max={seq_repr.max().item():.4f}, mean={seq_repr.mean().item():.4f}")
                    else:
                        # 处理非tensor类型的输出
                        print(f"[数据集调试] 索引 {index} - ESM3编码结果为非tensor类型: {type(seq_repr)}")
                        sequence_embedding = torch.tensor(seq_repr, dtype=torch.float32)
                        if sequence_embedding.dim() > 1:
                            sequence_embedding = sequence_embedding.mean(dim=0)
                else:
                    print(f"[数据集调试] 索引 {index} - ESM3编码结果无sequence属性，使用零向量")
                    # 使用合理的默认嵌入维度
                    sequence_embedding = torch.zeros(2560, dtype=torch.float32)  # ESM3的典型输出维度
            else:
                print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                print(f"[数据集调试] 索引 {index} - ⚠️ ESM3模型未设置，无法进行编码")
                # 返回原始序列，让模型处理
                sequence_embedding = seq
        except Exception as e:
            print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            print(f"[数据集调试] 索引 {index} - ❌ ESM3编码失败: {str(e)}")
            # 发生错误时返回原始序列
            sequence_embedding = seq
        
        if self.use_bias_feature:
            coords = {k: v[:self.max_length] for k, v in entry['coords'].items()}
        else:
            coords = None

        label = entry["label"] if self.preset_label is None else self.preset_label

        # 返回编码后的嵌入或原始序列（如果编码失败）
        return sequence_embedding, label, coords

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        embeddings, label_ids, coords = tuple(zip(*batch))
        
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}

        # 检查第一个元素的类型来决定如何处理
        first_embedding = embeddings[0]
        
        if torch.is_tensor(first_embedding):
            # 所有输入都是编码后的嵌入张量
            print(f"[数据集调试] 批处理大小: {len(embeddings)}, 嵌入维度: {first_embedding.shape}")
            
            # 堆叠所有嵌入
            try:
                stacked_embeddings = torch.stack(embeddings)
                print(f"[数据集调试] 堆叠后的嵌入形状: {stacked_embeddings.shape}")
                inputs = {"embeddings": stacked_embeddings}
            except Exception as e:
                print(f"[数据集调试] ❌ 堆叠嵌入失败: {str(e)}")
                # 回退到序列处理
                inputs = {"sequences": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings]}
        else:
            # 包含原始序列（编码失败的情况）
            print(f"[数据集调试] 批处理包含原始序列，将由模型处理")
            inputs = {"sequences": embeddings}

        if self.use_bias_feature:
            inputs["coords"] = coords

        return inputs, labels