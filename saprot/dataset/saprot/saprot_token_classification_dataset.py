import json
import random
import torch

from ..data_interface import register_dataset
from transformers import AutoTokenizer, EsmTokenizer
from ..lmdb_dataset import *
from data.data_transform import pad_sequences
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein


@register_dataset
class SaprotTokenClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str = None,  # Keep parameter for compatibility but not used for ESM3
                 max_length: int = 1024,
                 fixed_seq_length: int = 2048,  # 添加固定序列长度参数
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer (not used for ESM3, kept for compatibility)
            max_length: Max length of sequence
            fixed_seq_length: 固定序列长度，用于截断或padding
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
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer) if not isinstance(tokenizer, str) or 'esm3' not in tokenizer.lower() else None
            self.is_saprot_model = 'saprot' in tokenizer.lower() if isinstance(tokenizer, str) else False
            self.is_esm3_model = 'esm3' in tokenizer.lower() if isinstance(tokenizer, str) else False
        else:
            self.tokenizer = None
            self.is_saprot_model = False  # ESM3 doesn't need saprot tokenizer
            self.is_esm3_model = True  # Default to ESM3 when no tokenizer provided
        
        self.max_length = max_length
        self.fixed_seq_length = fixed_seq_length

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

    def _pad_or_truncate_list(self, lst, target_length, pad_value=-1):
        """
        将列表截断或padding到固定长度
        Args:
            lst: 输入列表
            target_length: 目标长度
            pad_value: padding值
        Returns:
            处理后的列表
        """
        if len(lst) > target_length:
            # 截断
            return lst[:target_length]
        elif len(lst) < target_length:
            # padding
            return lst + [pad_value] * (target_length - len(lst))
        else:
            return lst

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length-2]
        label = entry["label"][:self.max_length-2]

        # Convert sequence to string format for ESM3
        if isinstance(seq, list):
            seq = "".join(seq)

        if self.is_esm3_model:
            # 在主线程中进行ESM3编码
            try:
                if self.esm_model is not None:
                    # 使用ESM3模型编码sequence
                    # print(f"[token分类数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                    
                    # 创建ESMProtein对象并编码
                    protein = ESMProtein(sequence=seq)
                    
                    with torch.no_grad():  # 编码时不需要梯度
                        try:
                            # 直接使用encode方法获取encoded_protein
                            encoded_protein = self.esm_model.encode(protein)
                            # print(f"[token分类数据集调试] 索引 {index} - ✅ ESM3编码成功，类型: {type(encoded_protein)}")
                            
                            # 从encoded_protein中提取sequence token
                            if hasattr(encoded_protein, 'sequence'):
                                sequence_tokens = getattr(encoded_protein, 'sequence')
                                # print(f"[token分类数据集调试] 索引 {index} - 提取到sequence tokens，类型: {type(sequence_tokens)}")
                                
                                if torch.is_tensor(sequence_tokens):
                                    # print(f"[token分类数据集调试] 索引 {index} - Token形状: {sequence_tokens.shape}, dtype: {sequence_tokens.dtype}, device: {sequence_tokens.device}")
                                    # 将tensor移动到模型设备（通常是GPU）
                                    sequence_tokens = sequence_tokens.to(self.model_device)
                                    # 直接在数据集中进行固定长度处理
                                    sequence_embedding = self._pad_or_truncate_tensor(sequence_tokens, self.fixed_seq_length)
                                    # print(f"[token分类数据集调试] 索引 {index} - 固定长度后形状: {sequence_embedding.shape}, device: {sequence_embedding.device}")
                                else:
                                    # 如果不是tensor，转换为tensor并处理
                                    # 在模型设备上创建tensor（GPU训练时直接在GPU上）
                                    sequence_tokens = torch.tensor(sequence_tokens, device=self.model_device)
                                    sequence_embedding = self._pad_or_truncate_tensor(sequence_tokens, self.fixed_seq_length)
                                    # print(f"[token分类数据集调试] 索引 {index} - 转换并固定长度后形状: {sequence_embedding.shape}, device: {sequence_embedding.device}")
                            else:
                                # print(f"[token分类数据集调试] 索引 {index} - encoded_protein没有sequence属性")
                                # print(f"[token分类数据集调试] 索引 {index} - encoded_protein属性: {[attr for attr in dir(encoded_protein) if not attr.startswith('_')]}")
                                # 返回原始序列
                                sequence_embedding = seq
                                
                        except Exception as encode_error:
                            # print(f"[token分类数据集调试] 索引 {index} - ESM3编码失败: {str(encode_error)}")
                            # 发生错误时返回原始序列
                            sequence_embedding = seq
                else:
                    # print(f"[token分类数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                    # print(f"[token分类数据集调试] 索引 {index} - ⚠️ ESM3模型未设置，无法进行编码")
                    # 返回原始序列，让模型处理
                    sequence_embedding = seq
            except Exception as e:
                # print(f"[token分类数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                # print(f"[token分类数据集调试] 索引 {index} - ❌ ESM3编码失败: {str(e)}")
                # 发生错误时返回原始序列
                sequence_embedding = seq
            
            # 对于ESM3模型，处理标签长度以匹配固定序列长度
            if torch.is_tensor(sequence_embedding):
                # 如果sequence_embedding是tensor，标签也需要对应调整
                label = self._pad_or_truncate_list(label, self.fixed_seq_length, -1)
                label = torch.tensor(label, dtype=torch.long, device=self.model_device)
            else:
                # 如果是原始序列，直接使用原始标签
                label = torch.tensor(label, dtype=torch.long)
                
            return sequence_embedding, label
        elif self.is_saprot_model:
            processed_seq = []
            for aa in seq:
                processed_seq.append(aa + "#")
            seq = processed_seq
            seq = " ".join(seq)
            tokens = self.tokenizer.tokenize(seq)[:self.max_length]
            seq = " ".join(tokens)
            # Add -1 to the start and end of the label to ignore the cls token
            label = [-1] + label + [-1]
            label = torch.tensor(label, dtype=torch.long)
            return seq, label
        else:
            seq = " ".join(seq)
            tokens = self.tokenizer.tokenize(seq)[:self.max_length]
            seq = " ".join(tokens)
            # Add -1 to the start and end of the label to ignore the cls token
            label = [-1] + label + [-1]
            label = torch.tensor(label, dtype=torch.long)
            return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids = tuple(zip(*batch))

        if self.is_esm3_model:
            # 检查第一个元素的类型来决定如何处理
            first_seq = seqs[0]
            
            if torch.is_tensor(first_seq):
                # 所有输入都是token tensor，且应该已经是固定长度
                # print(f"[token分类数据集调试] 批处理大小: {len(seqs)}, 固定token长度: {first_seq.shape}")
                
                # 验证所有tensor都是相同长度
                expected_length = self.fixed_seq_length
                processed_tokens = []
                processed_labels = []
                
                for i, (seq_emb, label) in enumerate(zip(seqs, label_ids)):
                    if torch.is_tensor(seq_emb):
                        if seq_emb.shape[0] != expected_length:
                            # print(f"[token分类数据集调试] ⚠️ 样本 {i} 长度不匹配: {seq_emb.shape[0]} vs {expected_length}，重新处理")
                            # 重新进行截断或padding
                            seq_emb = self._pad_or_truncate_tensor(seq_emb, expected_length)
                        processed_tokens.append(seq_emb)
                    else:
                        # 创建固定长度的零tensor，使用与第一个tensor相同的设备
                        # print(f"[token分类数据集调试] ⚠️ 样本 {i} 不是tensor，创建零tensor")
                        device = processed_tokens[0].device if processed_tokens else self.model_device
                        processed_tokens.append(torch.zeros(expected_length, dtype=torch.long, device=device))
                    
                    # 处理标签
                    if torch.is_tensor(label):
                        if label.shape[0] != expected_length:
                            # 重新进行截断或padding
                            if len(label) > expected_length:
                                label = label[:expected_length]
                            elif len(label) < expected_length:
                                padding_size = expected_length - len(label)
                                padding = torch.full((padding_size,), -1, dtype=label.dtype, device=label.device)
                                label = torch.cat([label, padding])
                        processed_labels.append(label)
                    else:
                        # 创建固定长度的标签tensor
                        device = processed_tokens[0].device if processed_tokens else self.model_device
                        processed_labels.append(torch.full((expected_length,), -1, dtype=torch.long, device=device))
                
                try:
                    stacked_tokens = torch.stack(processed_tokens)
                    stacked_labels = torch.stack(processed_labels)
                    # print(f"[token分类数据集调试] 堆叠后的固定长度token形状: {stacked_tokens.shape}")
                    # print(f"[token分类数据集调试] 堆叠后的固定长度label形状: {stacked_labels.shape}")
                    inputs = {"tokens": stacked_tokens}
                    labels = {"labels": stacked_labels}
                except Exception as e:
                    # print(f"[token分类数据集调试] ❌ 堆叠tokens失败: {str(e)}")
                    # 回退到序列处理
                    inputs = {"sequences": [str(emb) if torch.is_tensor(emb) else emb for emb in seqs]}
                    label_ids = pad_sequences(label_ids, constant_value=-1)
                    labels = {"labels": label_ids}
            else:
                # 包含原始序列（编码失败的情况）
                # print(f"[token分类数据集调试] 批处理包含原始序列，将由模型处理")
                inputs = {"sequences": seqs}
                label_ids = pad_sequences(label_ids, constant_value=-1)
                labels = {"labels": label_ids}
            
            return inputs, labels
        else:
            # 对于其他模型，使用原有的处理逻辑
            label_ids = pad_sequences(label_ids, constant_value=-1)
            labels = {"labels": label_ids}

            encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
            inputs = {"inputs": encoder_info}
    
            return inputs, labels