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
        
        # 打印sequence及其ESM3编码token
        try:
            if self.esm_model is not None:
                # 使用ESM3模型编码sequence
                protein = ESMProtein(sequence=seq)
                encoded_protein = self.esm_model.encode(protein)
                
                # 获取编码token (这里我们获取sequence representation的维度信息)
                if hasattr(encoded_protein, 'sequence'):
                    seq_repr = encoded_protein.sequence
                    if torch.is_tensor(seq_repr):
                        token_info = f"Token tensor shape: {seq_repr.shape}, dtype: {seq_repr.dtype}"
                        print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                        print(f"[数据集调试] 索引 {index} - ESM3 编码: {token_info}")
                        print(f"[数据集调试] 索引 {index} - Token 统计: min={seq_repr.min().item():.4f}, max={seq_repr.max().item():.4f}, mean={seq_repr.mean().item():.4f}")
                    else:
                        print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                        print(f"[数据集调试] 索引 {index} - ESM3 编码: 非tensor类型 - {type(seq_repr)}")
                else:
                    print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                    print(f"[数据集调试] 索引 {index} - ESM3 编码: 无sequence属性")
            else:
                print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
                print(f"[数据集调试] 索引 {index} - ESM3 模型未设置，无法显示编码token")
        except Exception as e:
            print(f"[数据集调试] 索引 {index} - Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
            print(f"[数据集调试] 索引 {index} - ESM3 编码失败: {str(e)}")
        
        if self.use_bias_feature:
            coords = {k: v[:self.max_length] for k, v in entry['coords'].items()}
        else:
            coords = None

        label = entry["label"] if self.preset_label is None else self.preset_label

        # Return raw sequence instead of encoded protein to avoid multiprocessing issues
        return seq, label, coords

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        sequences, label_ids, coords = tuple(zip(*batch))
        
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}
    
        # For ESM3 compatibility, just return the sequences
        # The model will handle ESM3 encoding
        inputs = {"sequences": sequences}

        if self.use_bias_feature:
            inputs["coords"] = coords

        return inputs, labels