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
                
                # 创建ESMProtein对象并编码
                protein = ESMProtein(sequence=seq)
                
                with torch.no_grad():  # 编码时不需要梯度
                    try:
                        # 直接使用encode方法获取encoded_protein
                        encoded_protein = self.esm_model.encode(protein)
                        print(f"[数据集调试] 索引 {index} - ✅ ESM3编码成功，类型: {type(encoded_protein)}")
                        
                        # 从encoded_protein中提取sequence token
                        if hasattr(encoded_protein, 'sequence'):
                            sequence_tokens = getattr(encoded_protein, 'sequence')
                            print(f"[数据集调试] 索引 {index} - 提取到sequence tokens，类型: {type(sequence_tokens)}")
                            
                            if torch.is_tensor(sequence_tokens):
                                print(f"[数据集调试] 索引 {index} - Token形状: {sequence_tokens.shape}, dtype: {sequence_tokens.dtype}")
                                sequence_embedding = sequence_tokens
                            else:
                                # 如果不是tensor，转换为tensor
                                sequence_embedding = torch.tensor(sequence_tokens)
                                print(f"[数据集调试] 索引 {index} - 转换为tensor，形状: {sequence_embedding.shape}")
                        else:
                            print(f"[数据集调试] 索引 {index} - encoded_protein没有sequence属性")
                            print(f"[数据集调试] 索引 {index} - encoded_protein属性: {[attr for attr in dir(encoded_protein) if not attr.startswith('_')]}")
                            # 返回原始序列
                            sequence_embedding = seq
                            
                    except Exception as encode_error:
                        print(f"[数据集调试] 索引 {index} - ESM3编码失败: {str(encode_error)}")
                        # 发生错误时返回原始序列
                        sequence_embedding = seq
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
            # 所有输入都是token tensor
            print(f"[数据集调试] 批处理大小: {len(embeddings)}, token形状: {first_embedding.shape}")
            
            # 检查是否需要padding（如果长度不同）
            max_len = max(emb.shape[0] if torch.is_tensor(emb) else 0 for emb in embeddings)
            print(f"[数据集调试] 最大序列长度: {max_len}")
            
            # 处理padding
            padded_tokens = []
            for emb in embeddings:
                if torch.is_tensor(emb):
                    if emb.shape[0] < max_len:
                        # 需要padding，用0填充
                        padding_size = max_len - emb.shape[0]
                        padded_emb = torch.cat([emb, torch.zeros(padding_size, dtype=emb.dtype)])
                        padded_tokens.append(padded_emb)
                    else:
                        padded_tokens.append(emb)
                else:
                    # 不是tensor的情况，创建零tensor
                    padded_tokens.append(torch.zeros(max_len, dtype=torch.long))
            
            try:
                stacked_tokens = torch.stack(padded_tokens)
                print(f"[数据集调试] 堆叠后的token形状: {stacked_tokens.shape}")
                inputs = {"tokens": stacked_tokens}
            except Exception as e:
                print(f"[数据集调试] ❌ 堆叠tokens失败: {str(e)}")
                # 回退到序列处理
                inputs = {"sequences": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings]}
        else:
            # 包含原始序列（编码失败的情况）
            print(f"[数据集调试] 批处理包含原始序列，将由模型处理")
            inputs = {"sequences": embeddings}

        if self.use_bias_feature:
            inputs["coords"] = coords

        return inputs, labels