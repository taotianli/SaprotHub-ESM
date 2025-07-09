import torch
import json

from ..lmdb_dataset import LMDBDataset
from transformers import AutoTokenizer, EsmTokenizer
from ..data_interface import register_dataset
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


@register_dataset
class SaprotPairClassificationDataset(LMDBDataset):
    def __init__(self,
             tokenizer: str = None,  # Keep parameter for compatibility but not used for ESM3
             max_length: int = 1024,
             fixed_seq_length: int = 2048,  # 添加固定序列长度参数
             plddt_threshold: float = None,
             **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer (not used for ESM3, kept for compatibility)
            
            max_length: Max length of sequence
            
            fixed_seq_length: 固定序列长度，用于截断或padding
            
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
        
        # Only initialize tokenizer if provided and not using ESM3
        if tokenizer is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.is_saprot_model = 'saprot' in tokenizer.lower() if isinstance(tokenizer, str) else False
        else:
            self.tokenizer = None
            self.is_saprot_model = False  # ESM3 doesn't need saprot tokenizer
        
        self.max_length = max_length
        self.fixed_seq_length = fixed_seq_length
        self.plddt_threshold = plddt_threshold

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
        seq_1, seq_2 = entry['seq_1'][:self.max_length-2], entry['seq_2'][:self.max_length-2]

        # Convert sequences to string format for ESM3
        if isinstance(seq_1, list):
            seq_1 = "".join(seq_1)
        if isinstance(seq_2, list):
            seq_2 = "".join(seq_2)
        
        # 在主线程中进行ESM3编码
        try:
            if self.esm_model is not None:
                # 使用ESM3模型编码sequence对
                # print(f"[pair分类数据集调试] 索引 {index} - Sequence1: {seq_1[:50]}{'...' if len(seq_1) > 50 else ''}")
                # print(f"[pair分类数据集调试] 索引 {index} - Sequence2: {seq_2[:50]}{'...' if len(seq_2) > 50 else ''}")
                
                # 创建ESMProtein对象并编码
                protein_1 = ESMProtein(sequence=seq_1)
                protein_2 = ESMProtein(sequence=seq_2)
                
                with torch.no_grad():  # 编码时不需要梯度
                    try:
                        # 直接使用encode方法获取encoded_protein
                        encoded_protein_1 = self.esm_model.encode(protein_1)
                        encoded_protein_2 = self.esm_model.encode(protein_2)
                        # print(f"[pair分类数据集调试] 索引 {index} - ✅ ESM3编码成功")
                        
                        # 从encoded_protein中提取sequence token
                        if hasattr(encoded_protein_1, 'sequence') and hasattr(encoded_protein_2, 'sequence'):
                            sequence_tokens_1 = getattr(encoded_protein_1, 'sequence')
                            sequence_tokens_2 = getattr(encoded_protein_2, 'sequence')
                            
                            if torch.is_tensor(sequence_tokens_1) and torch.is_tensor(sequence_tokens_2):
                                # 将tensor移动到模型设备（通常是GPU）
                                sequence_tokens_1 = sequence_tokens_1.to(self.model_device)
                                sequence_tokens_2 = sequence_tokens_2.to(self.model_device)
                                # 直接在数据集中进行固定长度处理
                                sequence_embedding_1 = self._pad_or_truncate_tensor(sequence_tokens_1, self.fixed_seq_length)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(sequence_tokens_2, self.fixed_seq_length)
                            else:
                                # 如果不是tensor，转换为tensor并处理
                                sequence_tokens_1 = torch.tensor(sequence_tokens_1, device=self.model_device)
                                sequence_tokens_2 = torch.tensor(sequence_tokens_2, device=self.model_device)
                                sequence_embedding_1 = self._pad_or_truncate_tensor(sequence_tokens_1, self.fixed_seq_length)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(sequence_tokens_2, self.fixed_seq_length)
                        else:
                            # 返回原始序列
                            sequence_embedding_1 = seq_1
                            sequence_embedding_2 = seq_2
                            
                    except Exception as encode_error:
                        # print(f"[pair分类数据集调试] 索引 {index} - ESM3编码失败: {str(encode_error)}")
                        # 发生错误时返回原始序列
                        sequence_embedding_1 = seq_1
                        sequence_embedding_2 = seq_2
            else:
                # print(f"[pair分类数据集调试] 索引 {index} - ⚠️ ESM3模型未设置，使用传统tokenizer处理")
                
                # 使用传统tokenizer处理 (如果有的话)
                if self.tokenizer is not None:
                    if self.is_saprot_model:
                        processed_seq_1 = []
                        processed_seq_2 = []
                        for aa in seq_1:
                            processed_seq_1.append(aa + "#")
                        seq_1 = processed_seq_1
                        for aa in seq_2:
                            processed_seq_2.append(aa + "#")
                        seq_2 = processed_seq_2
                        
                    seq_1 = " ".join(seq_1)
                    seq_2 = " ".join(seq_2)
                    
                    # Mask structure tokens with pLDDT < threshold
                    if self.plddt_threshold is not None:
                        plddt_1, plddt_2 = entry['plddt_1'], entry['plddt_2']
                        tokens = self.tokenizer.tokenize(seq_1)
                        seq_1 = ""
                        assert len(tokens) == len(plddt_1)
                        for token, score in zip(tokens, plddt_1):
                            if score < self.plddt_threshold:
                                seq_1 += token[:-1] + "#"
                            else:
                                seq_1 += token
                        
                        tokens = self.tokenizer.tokenize(seq_2)
                        seq_2 = ""
                        assert len(tokens) == len(plddt_2)
                        for token, score in zip(tokens, plddt_2):
                            if score < self.plddt_threshold:
                                seq_2 += token[:-1] + "#"
                            else:
                                seq_2 += token
                                
                    tokens = self.tokenizer.tokenize(seq_1)[:self.max_length]
                    seq_1 = " ".join(tokens)

                    tokens = self.tokenizer.tokenize(seq_2)[:self.max_length]
                    seq_2 = " ".join(seq_2)
                
                # 返回原始序列，让模型处理
                sequence_embedding_1 = seq_1
                sequence_embedding_2 = seq_2
        except Exception as e:
            # print(f"[pair分类数据集调试] 索引 {index} - ❌ 处理失败: {str(e)}")
            # 发生错误时返回原始序列
            sequence_embedding_1 = seq_1
            sequence_embedding_2 = seq_2

        return sequence_embedding_1, sequence_embedding_2, int(entry["label"])

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        embeddings_1, embeddings_2, label_ids = tuple(zip(*batch))
        
        label_ids = torch.tensor(label_ids, dtype=torch.long, device=self.model_device)
        labels = {"labels": label_ids}

        # 检查第一个元素的类型来决定如何处理
        first_embedding_1 = embeddings_1[0]
        first_embedding_2 = embeddings_2[0]
        
        if torch.is_tensor(first_embedding_1) and torch.is_tensor(first_embedding_2):
            # 所有输入都是token tensor，且应该已经是固定长度
            # print(f"[pair分类数据集调试] 批处理大小: {len(embeddings_1)}, 固定token长度: {first_embedding_1.shape}")
            
            # 验证所有tensor都是相同长度
            expected_length = self.fixed_seq_length
            processed_tokens_1 = []
            processed_tokens_2 = []
            
            for i, (emb_1, emb_2) in enumerate(zip(embeddings_1, embeddings_2)):
                if torch.is_tensor(emb_1) and torch.is_tensor(emb_2):
                    if emb_1.shape[0] != expected_length:
                        # 重新进行截断或padding
                        emb_1 = self._pad_or_truncate_tensor(emb_1, expected_length)
                    if emb_2.shape[0] != expected_length:
                        # 重新进行截断或padding
                        emb_2 = self._pad_or_truncate_tensor(emb_2, expected_length)
                    processed_tokens_1.append(emb_1)
                    processed_tokens_2.append(emb_2)
                else:
                    # 创建固定长度的零tensor，使用与第一个tensor相同的设备
                    device = processed_tokens_1[0].device if processed_tokens_1 else self.model_device
                    processed_tokens_1.append(torch.zeros(expected_length, dtype=torch.long, device=device))
                    processed_tokens_2.append(torch.zeros(expected_length, dtype=torch.long, device=device))
            
            try:
                stacked_tokens_1 = torch.stack(processed_tokens_1)
                stacked_tokens_2 = torch.stack(processed_tokens_2)
                # print(f"[pair分类数据集调试] 堆叠后的固定长度token形状: {stacked_tokens_1.shape}, {stacked_tokens_2.shape}")
                inputs = {"tokens_1": stacked_tokens_1, "tokens_2": stacked_tokens_2}
            except Exception as e:
                # print(f"[pair分类数据集调试] ❌ 堆叠tokens失败: {str(e)}")
                # 回退到序列处理
                inputs = {"sequences_1": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings_1],
                         "sequences_2": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings_2]}
        
        elif self.tokenizer is not None:
            # 使用传统tokenizer处理
            # print(f"[pair分类数据集调试] 使用传统tokenizer处理批处理")
            encoder_info_1 = self.tokenizer.batch_encode_plus(embeddings_1, return_tensors='pt', padding=True)
            encoder_info_2 = self.tokenizer.batch_encode_plus(embeddings_2, return_tensors='pt', padding=True)
            inputs = {"inputs_1": encoder_info_1, "inputs_2": encoder_info_2}
        else:
            # 包含原始序列（编码失败的情况）
            # print(f"[pair分类数据集调试] 批处理包含原始序列，将由模型处理")
            inputs = {"sequences_1": embeddings_1, "sequences_2": embeddings_2}

        return inputs, labels