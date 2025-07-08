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
                 fixed_seq_length: int = 2048,  # 添加固定序列长度参数
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
            fixed_seq_length: 固定序列长度，用于截断或padding
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
        self.model_device = 'cpu'  # 默认CPU，会在set_esm_model时更新
        
        self.max_length = max_length
        self.fixed_seq_length = fixed_seq_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self.mask_struc_ratio = mask_struc_ratio
        self.mask_seed = mask_seed
        self.plddt_threshold = plddt_threshold

        self.is_saprot_model = True  # Always true for ESM3

    def set_esm_model(self, esm_model):
        """Set the ESM3 model for encoding. This should be called from the main process."""
        self.esm_model = esm_model
        # 获取模型设备，用于确定返回tensor的设备
        self.model_device = next(esm_model.parameters()).device if esm_model is not None else 'cpu'

    def _pad_or_truncate_tensor(self, tensor, target_length):
        """
        将tensor截断或padding到固定长度
        Args:
            tensor: 输入tensor [seq_len] 
            target_length: 目标长度
        Returns:
            处理后的tensor [target_length]
        """
        if len(tensor) > target_length:
            # 截断
            return tensor[:target_length]
        elif len(tensor) < target_length:
            # padding - 确保padding tensor和原tensor在同一设备上
            padding_size = target_length - len(tensor)
            padding = torch.zeros(padding_size, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, padding])
        else:
            return tensor

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
                                print(f"[数据集调试] 索引 {index} - Token形状: {sequence_tokens.shape}, dtype: {sequence_tokens.dtype}, device: {sequence_tokens.device}")
                                # 将tensor移动到模型设备（通常是GPU）
                                sequence_tokens = sequence_tokens.to(self.model_device)
                                # 直接在数据集中进行固定长度处理
                                sequence_embedding = self._pad_or_truncate_tensor(sequence_tokens, self.fixed_seq_length)
                                print(f"[数据集调试] 索引 {index} - 固定长度后形状: {sequence_embedding.shape}, device: {sequence_embedding.device}")
                            else:
                                # 如果不是tensor，转换为tensor并处理
                                # 在模型设备上创建tensor（GPU训练时直接在GPU上）
                                sequence_tokens = torch.tensor(sequence_tokens, device=self.model_device)
                                sequence_embedding = self._pad_or_truncate_tensor(sequence_tokens, self.fixed_seq_length)
                                print(f"[数据集调试] 索引 {index} - 转换并固定长度后形状: {sequence_embedding.shape}, device: {sequence_embedding.device}")
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
        
        label_ids = torch.tensor(label_ids, dtype=torch.long, device=self.model_device)
        labels = {"labels": label_ids}

        # 检查第一个元素的类型来决定如何处理
        first_embedding = embeddings[0]
        
        if torch.is_tensor(first_embedding):
            # 所有输入都是token tensor，且应该已经是固定长度
            print(f"[数据集调试] 批处理大小: {len(embeddings)}, 固定token长度: {first_embedding.shape}")
            
            # 验证所有tensor都是相同长度
            expected_length = self.fixed_seq_length
            processed_tokens = []
            
            for i, emb in enumerate(embeddings):
                if torch.is_tensor(emb):
                    if emb.shape[0] != expected_length:
                        print(f"[数据集调试] ⚠️ 样本 {i} 长度不匹配: {emb.shape[0]} vs {expected_length}，重新处理")
                        # 重新进行截断或padding
                        emb = self._pad_or_truncate_tensor(emb, expected_length)
                    processed_tokens.append(emb)
                else:
                    # 创建固定长度的零tensor，使用与第一个tensor相同的设备
                    print(f"[数据集调试] ⚠️ 样本 {i} 不是tensor，创建零tensor")
                    device = processed_tokens[0].device if processed_tokens else self.model_device
                    processed_tokens.append(torch.zeros(expected_length, dtype=torch.long, device=device))
            
            try:
                stacked_tokens = torch.stack(processed_tokens)
                print(f"[数据集调试] 堆叠后的固定长度token形状: {stacked_tokens.shape}")
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