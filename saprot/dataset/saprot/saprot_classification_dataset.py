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
                
                # 创建ESMProtein对象
                protein = ESMProtein(sequence=seq)
                
                with torch.no_grad():  # 编码时不需要梯度
                    try:
                        # 方法1: 尝试使用encode方法然后forward获取embeddings
                        try:
                            # 首先编码protein
                            encoded_protein = self.esm_model.encode(protein)
                            
                            # 然后forward获取真正的嵌入
                            output = self.esm_model.forward(encoded_protein)
                            
                            # 查找嵌入属性
                            embedding = None
                            if hasattr(output, 'embeddings'):
                                embedding = output.embeddings
                                print(f"[数据集调试] 索引 {index} - 使用output.embeddings")
                            elif hasattr(output, 'last_hidden_state'):
                                embedding = output.last_hidden_state  
                                print(f"[数据集调试] 索引 {index} - 使用output.last_hidden_state")
                            elif hasattr(output, 'sequence_embeddings'):
                                embedding = output.sequence_embeddings
                                print(f"[数据集调试] 索引 {index} - 使用output.sequence_embeddings")
                            else:
                                print(f"[数据集调试] 索引 {index} - 无法找到嵌入属性，output属性: {dir(output)}")
                                
                        except Exception as encode_error:
                            print(f"[数据集调试] 索引 {index} - encode+forward方法失败: {str(encode_error)}")
                            embedding = None
                        
                        # 方法2: 如果方法1失败，尝试直接forward protein
                        if embedding is None:
                            try:
                                print(f"[数据集调试] 索引 {index} - 尝试直接forward protein")
                                output = self.esm_model.forward(protein)
                                
                                if hasattr(output, 'embeddings'):
                                    embedding = output.embeddings
                                    print(f"[数据集调试] 索引 {index} - 直接forward使用embeddings")
                                elif hasattr(output, 'last_hidden_state'):
                                    embedding = output.last_hidden_state
                                    print(f"[数据集调试] 索引 {index} - 直接forward使用last_hidden_state")
                                else:
                                    print(f"[数据集调试] 索引 {index} - 直接forward也无法找到嵌入，output属性: {dir(output)}")
                                    
                            except Exception as forward_error:
                                print(f"[数据集调试] 索引 {index} - 直接forward也失败: {str(forward_error)}")
                                embedding = None

                        # 方法3: 如果都失败，尝试使用模型的其他方法
                        if embedding is None:
                            try:
                                # 检查模型是否有其他编码方法
                                if hasattr(self.esm_model, 'encode_sequence'):
                                    embedding = self.esm_model.encode_sequence(seq)
                                    print(f"[数据集调试] 索引 {index} - 使用encode_sequence方法")
                                elif hasattr(self.esm_model, 'get_embeddings'):
                                    embedding = self.esm_model.get_embeddings(protein)
                                    print(f"[数据集调试] 索引 {index} - 使用get_embeddings方法")
                                else:
                                    print(f"[数据集调试] 索引 {index} - 模型方法: {[m for m in dir(self.esm_model) if not m.startswith('_')]}")
                                    embedding = None
                                    
                            except Exception as alt_error:
                                print(f"[数据集调试] 索引 {index} - 替代方法也失败: {str(alt_error)}")
                                embedding = None

                        # 处理获得的嵌入
                        if embedding is not None and torch.is_tensor(embedding):
                            # 确保数据类型为float
                            if embedding.dtype in [torch.int64, torch.int32, torch.long]:
                                print(f"[数据集调试] 索引 {index} - 警告: 获得的是整数类型 {embedding.dtype}，可能是token IDs而非嵌入")
                                # 这种情况下我们无法使用，返回原始序列
                                sequence_embedding = seq
                            else:
                                # 应用平均池化获得固定长度的representation
                                embedding = embedding.float()  # 确保是float类型
                                
                                print(f"[数据集调试] 索引 {index} - ESM3原始输出形状: {embedding.shape}, dtype: {embedding.dtype}")
                                
                                # 处理不同维度的嵌入
                                if embedding.dim() == 3:  # [batch, seq_len, hidden_dim]
                                    if embedding.shape[0] == 1:  # batch=1
                                        embedding = embedding.squeeze(0)  # [seq_len, hidden_dim]
                                    sequence_embedding = embedding.mean(dim=0)  # [hidden_dim]
                                elif embedding.dim() == 2:  # [seq_len, hidden_dim] 
                                    sequence_embedding = embedding.mean(dim=0)  # [hidden_dim]
                                elif embedding.dim() == 1:  # [hidden_dim]
                                    sequence_embedding = embedding
                                else:
                                    print(f"[数据集调试] 索引 {index} - 未知的嵌入维度: {embedding.shape}")
                                    sequence_embedding = embedding.flatten()
                                
                                print(f"[数据集调试] 索引 {index} - 池化后嵌入形状: {sequence_embedding.shape}")
                                print(f"[数据集调试] 索引 {index} - 嵌入统计: min={embedding.min().item():.4f}, max={embedding.max().item():.4f}, mean={embedding.mean().item():.4f}")
                                print(f"[数据集调试] 索引 {index} - ✅ ESM3编码成功")
                        else:
                            print(f"[数据集调试] 索引 {index} - 所有方法都失败，使用零向量")
                            sequence_embedding = torch.zeros(2560, dtype=torch.float32)
                            
                    except Exception as outer_error:
                        print(f"[数据集调试] 索引 {index} - 外层编码失败: {str(outer_error)}")
                        sequence_embedding = torch.zeros(2560, dtype=torch.float32)
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