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
        super().__init__(**kwargs)
        # Initialize ESM3 model for encoding
        self.esm_model = ESM3.from_pretrained("esm3-open")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.esm_model = self.esm_model.to(device)
        
        self.max_length = max_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self.mask_struc_ratio = mask_struc_ratio
        self.mask_seed = mask_seed
        self.plddt_threshold = plddt_threshold

        self.is_saprot_model = True  # Always true for ESM3

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][:self.max_length-2]
        
        # Convert sequence to string format for ESM3
        if isinstance(seq, list):
            seq = "".join(seq)
        
        # Apply masking if needed (simplified for ESM3)
        if self.mask_struc_ratio is not None:
            # Simple random masking for compatibility
            random.seed(self.mask_seed)
            seq_list = list(seq)
            mask_num = int(len(seq_list) * self.mask_struc_ratio)
            mask_indices = random.sample(range(len(seq_list)), mask_num)
            for idx in mask_indices:
                seq_list[idx] = 'X'  # Use X for masked tokens
            seq = "".join(seq_list)
        
        # Create ESMProtein object and encode
        protein = ESMProtein(sequence=seq)
        with torch.no_grad():
            encoded_protein = self.esm_model.encode(protein)
        
        if self.use_bias_feature:
            coords = {k: v[:self.max_length] for k, v in entry['coords'].items()}
        else:
            coords = None

        label = entry["label"] if self.preset_label is None else self.preset_label

        return encoded_protein, label, coords

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        encoded_proteins, label_ids, coords = tuple(zip(*batch))
        
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}
    
        # Process ESM3 encoded proteins - simplified approach
        # For now, just pass the encoded proteins directly
        # The model will handle the extraction of features
        inputs = {"inputs": encoded_proteins}

        if self.use_bias_feature:
            inputs["coords"] = coords

        return inputs, labels