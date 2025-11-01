import torch
import json

from ..lmdb_dataset import LMDBDataset
# transformers的tokenizer对ESM3不需要，ESM3使用自己的编码方式
# from transformers import AutoTokenizer, EsmTokenizer
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
        
        self.max_length = max_length
        self.fixed_seq_length = fixed_seq_length
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
        seq_1, seq_2 = entry['seq_1'][:self.max_length-2], entry['seq_2'][:self.max_length-2]

        # 检查是否有pdb_path字段（结构数据）
        has_structure_1 = 'pdb_path_1' in entry and entry['pdb_path_1'] is not None
        has_structure_2 = 'pdb_path_2' in entry and entry['pdb_path_2'] is not None

        # Convert sequences to string format for ESM3
        if isinstance(seq_1, list):
            seq_1 = "".join(seq_1)
        if isinstance(seq_2, list):
            seq_2 = "".join(seq_2)
        
        # 在主线程中进行ESM3编码
        try:
            if self.esm_model is not None:
                # 处理第一个蛋白
                if has_structure_1:
                    from esm.utils.structure.protein_chain import ProteinChain
                    
                    pdb_path_1 = entry['pdb_path_1']
                    # 从CSV中读取chain_1列，如果没有则默认为'A'
                    chain_id_1 = entry.get('chain_1', 'A')
                    
                    try:
                        # 使用ProteinChain读取PDB
                        chain_1 = ProteinChain.from_pdb(pdb_path_1, chain_id=chain_id_1)
                        
                        # 创建包含结构的ESMProtein对象
                        protein_1 = ESMProtein(
                            sequence=chain_1.sequence,
                            coordinates=chain_1.atom37_positions
                        )
                    except Exception as pdb_error:
                        print(f"[Pair数据集警告] 索引 {index} - 读取PDB1失败: {str(pdb_error)}，回退到序列模式")
                        protein_1 = ESMProtein(sequence=seq_1)
                        has_structure_1 = False
                else:
                    protein_1 = ESMProtein(sequence=seq_1)
                
                # 处理第二个蛋白
                if has_structure_2:
                    from esm.utils.structure.protein_chain import ProteinChain
                    
                    pdb_path_2 = entry['pdb_path_2']
                    # 从CSV中读取chain_2列，如果没有则默认为'A'
                    chain_id_2 = entry.get('chain_2', 'A')
                    
                    try:
                        # 使用ProteinChain读取PDB
                        chain_2 = ProteinChain.from_pdb(pdb_path_2, chain_id=chain_id_2)
                        
                        # 创建包含结构的ESMProtein对象
                        protein_2 = ESMProtein(
                            sequence=chain_2.sequence,
                            coordinates=chain_2.atom37_positions
                        )
                    except Exception as pdb_error:
                        print(f"[Pair数据集警告] 索引 {index} - 读取PDB2失败: {str(pdb_error)}，回退到序列模式")
                        protein_2 = ESMProtein(sequence=seq_2)
                        has_structure_2 = False
                else:
                    protein_2 = ESMProtein(sequence=seq_2)
                
                with torch.no_grad():  # 编码时不需要梯度
                    try:
                        # 直接使用encode方法获取encoded_protein
                        encoded_protein_1 = self.esm_model.encode(protein_1)
                        encoded_protein_2 = self.esm_model.encode(protein_2)
                        # print(f"[pair分类数据集调试] 索引 {index} - ESM3编码成功")
                        
                        # 处理第一个蛋白的tokens - 如果有结构，优先使用structure token
                        if has_structure_1 and hasattr(encoded_protein_1, 'structure'):
                            structure_tokens_1 = getattr(encoded_protein_1, 'structure')
                            if torch.is_tensor(structure_tokens_1):
                                structure_tokens_1 = structure_tokens_1.to(self.model_device)
                                sequence_embedding_1 = self._pad_or_truncate_tensor(structure_tokens_1, self.fixed_seq_length)
                            else:
                                structure_tokens_1 = torch.tensor(structure_tokens_1, device=self.model_device)
                                sequence_embedding_1 = self._pad_or_truncate_tensor(structure_tokens_1, self.fixed_seq_length)
                        elif hasattr(encoded_protein_1, 'sequence'):
                            sequence_tokens_1 = getattr(encoded_protein_1, 'sequence')
                            if torch.is_tensor(sequence_tokens_1):
                                sequence_tokens_1 = sequence_tokens_1.to(self.model_device)
                                sequence_embedding_1 = self._pad_or_truncate_tensor(sequence_tokens_1, self.fixed_seq_length)
                            else:
                                sequence_tokens_1 = torch.tensor(sequence_tokens_1, device=self.model_device)
                                sequence_embedding_1 = self._pad_or_truncate_tensor(sequence_tokens_1, self.fixed_seq_length)
                        else:
                            sequence_embedding_1 = seq_1
                        
                        # 处理第二个蛋白的tokens - 如果有结构，优先使用structure token
                        if has_structure_2 and hasattr(encoded_protein_2, 'structure'):
                            structure_tokens_2 = getattr(encoded_protein_2, 'structure')
                            if torch.is_tensor(structure_tokens_2):
                                structure_tokens_2 = structure_tokens_2.to(self.model_device)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(structure_tokens_2, self.fixed_seq_length)
                            else:
                                structure_tokens_2 = torch.tensor(structure_tokens_2, device=self.model_device)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(structure_tokens_2, self.fixed_seq_length)
                        elif hasattr(encoded_protein_2, 'sequence'):
                            sequence_tokens_2 = getattr(encoded_protein_2, 'sequence')
                            if torch.is_tensor(sequence_tokens_2):
                                sequence_tokens_2 = sequence_tokens_2.to(self.model_device)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(sequence_tokens_2, self.fixed_seq_length)
                            else:
                                sequence_tokens_2 = torch.tensor(sequence_tokens_2, device=self.model_device)
                                sequence_embedding_2 = self._pad_or_truncate_tensor(sequence_tokens_2, self.fixed_seq_length)
                        else:
                            sequence_embedding_2 = seq_2
                            
                    except Exception as encode_error:
                        # print(f"[pair分类数据集调试] 索引 {index} - ESM3编码失败: {str(encode_error)}")
                        # 发生错误时返回原始序列
                        sequence_embedding_1 = seq_1
                        sequence_embedding_2 = seq_2
            else:
                # print(f"[pair分类数据集调试] 索引 {index} - ESM3模型未设置，使用传统tokenizer处理")
                
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
            # print(f"[pair分类数据集调试] 索引 {index} - 处理失败: {str(e)}")
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
                # print(f"[pair分类数据集调试] 堆叠tokens失败: {str(e)}")
                # 回退到序列处理
                inputs = {"sequences_1": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings_1],
                         "sequences_2": [str(emb) if torch.is_tensor(emb) else emb for emb in embeddings_2]}
        else:
            # 包含原始序列（编码失败的情况）
            # print(f"[pair分类数据集调试] 批处理包含原始序列，将由模型处理")
            inputs = {"sequences_1": embeddings_1, "sequences_2": embeddings_2}

        return inputs, labels