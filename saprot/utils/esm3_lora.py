#!/usr/bin/env python3
"""
ESM3 LoRA Utilities
Provides LoRA fine-tuning capabilities for ESM3 models
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Optional


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Linear Layer
    
    Args:
        original_layer: Original Linear layer
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA A and B matrices
        self.lora_A = nn.Parameter(torch.randn(r, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, r))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output: x @ A^T @ B^T
        # Ensure LoRA parameters have the same dtype as input
        lora_A = self.lora_A.to(dtype=x.dtype)
        lora_B = self.lora_B.to(dtype=x.dtype)
        lora_output = x @ lora_A.T @ lora_B.T * self.scaling
        lora_output = self.dropout(lora_output)
        
        return original_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into original weights"""
        if self.r > 0:
            delta_weight = self.lora_B @ self.lora_A * self.scaling
            self.original_layer.weight.data += delta_weight
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from original weights"""
        if self.r > 0:
            delta_weight = self.lora_B @ self.lora_A * self.scaling
            self.original_layer.weight.data -= delta_weight


class ESM3LoRAWrapper(nn.Module):
    """
    ESM3 Model LoRA Wrapper
    """
    
    def __init__(
        self,
        esm3_model,
        target_modules: List[str] = None,
        r: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: str = "none"
    ):
        super().__init__()
        
        self.esm3_model = esm3_model
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.bias = bias
        
        # Default target modules
        if target_modules is None:
            target_modules = [
                "attn.layernorm_qkv.1",
                "attn.out_proj",
                "ffn.1",
                "ffn.3",
                "geom_attn.proj",
                "geom_attn.out_proj",
            ]
        
        self.target_modules = target_modules
        self.lora_layers = {}
        self.lora_modules = nn.ModuleList()
        
        self._add_lora_layers()
        self._freeze_original_parameters()
        
        print(f"LoRA layers added to ESM3 model:")
        print(f"   Target modules: {len(self.lora_layers)}")
        print(f"   LoRA rank: {r}")
        print(f"   LoRA alpha: {alpha}")
        print(f"   Dropout: {dropout}")
    
    def _add_lora_layers(self):
        """Add LoRA layers to target modules"""
        
        def add_lora_to_module(parent_module, module_name, full_name):
            if hasattr(parent_module, module_name):
                module = getattr(parent_module, module_name)
                
                if isinstance(module, nn.Linear):
                    for target in self.target_modules:
                        if target in full_name:
                            # print(f"   Adding LoRA to: {full_name}")
                            lora_layer = LoRALinear(
                                module,
                                r=self.r,
                                alpha=self.alpha,
                                dropout=self.dropout
                            )
                            setattr(parent_module, module_name, lora_layer)
                            self.lora_layers[full_name] = lora_layer
                            self.lora_modules.append(lora_layer)
                            break
                
                elif hasattr(module, '__dict__'):
                    for child_name in dir(module):
                        if not child_name.startswith('_') and hasattr(module, child_name):
                            child_module = getattr(module, child_name)
                            if isinstance(child_module, (nn.Module, nn.Linear)):
                                child_full_name = f"{full_name}.{child_name}" if full_name else child_name
                                add_lora_to_module(module, child_name, child_full_name)
        
        if hasattr(self.esm3_model, 'transformer'):
            transformer = self.esm3_model.transformer
            
            if hasattr(transformer, 'blocks'):
                for i, block in enumerate(transformer.blocks):
                    block_name = f"transformer.blocks.{i}"
                    
                    if hasattr(block, 'attn'):
                        add_lora_to_module(block, 'attn', f"{block_name}.attn")
                    
                    if hasattr(block, 'ffn'):
                        add_lora_to_module(block, 'ffn', f"{block_name}.ffn")
                    
                    if hasattr(block, 'geom_attn'):
                        add_lora_to_module(block, 'geom_attn', f"{block_name}.geom_attn")
        
        if hasattr(self.esm3_model, 'output_heads'):
            output_heads = self.esm3_model.output_heads
            for head_name in dir(output_heads):
                if not head_name.startswith('_') and hasattr(output_heads, head_name):
                    head = getattr(output_heads, head_name)
                    if isinstance(head, nn.Sequential):
                        add_lora_to_module(output_heads, head_name, f"output_heads.{head_name}")
    
    def _freeze_original_parameters(self):
        """Freeze all original parameters, keep only LoRA parameters trainable"""
        print("❄️ Freezing original ESM3 parameters...")
        
        for name, param in self.esm3_model.named_parameters():
            param.requires_grad = False
        
        lora_param_count = 0
        for lora_layer in self.lora_modules:
            lora_layer.lora_A.requires_grad = True
            lora_layer.lora_B.requires_grad = True
            lora_param_count += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()
        
        print(f"Original parameters frozen, LoRA parameters: {lora_param_count:,}")
    
    def forward(self, *args, **kwargs):
        return self.esm3_model(*args, **kwargs)
    
    def encode(self, *args, **kwargs):
        return self.esm3_model.encode(*args, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.esm3_model, name)
    
    def get_lora_parameters(self) -> List[torch.Tensor]:
        params = []
        for lora_layer in self.lora_layers.values():
            params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return params
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        state_dict = {}
        for name, lora_layer in self.lora_layers.items():
            safe_name = name.replace('.', '_')
            state_dict[f"{safe_name}.lora_A"] = lora_layer.lora_A
            state_dict[f"{safe_name}.lora_B"] = lora_layer.lora_B
        return state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for name, lora_layer in self.lora_layers.items():
            safe_name = name.replace('.', '_')
            if f"{safe_name}.lora_A" in state_dict:
                lora_layer.lora_A.data = state_dict[f"{safe_name}.lora_A"]
            if f"{safe_name}.lora_B" in state_dict:
                lora_layer.lora_B.data = state_dict[f"{safe_name}.lora_B"]
    
    def save_lora_weights(self, path: str):
        torch.save(self.get_lora_state_dict(), path)
        print(f"LoRA weights saved to: {path}")
    
    def load_lora_weights(self, path: str):
        state_dict = torch.load(path, map_location='cpu')
        self.load_lora_state_dict(state_dict)
        print(f"LoRA weights loaded from {path}")
    
    def merge_and_save_model(self, path: str):
        for lora_layer in self.lora_layers.values():
            lora_layer.merge_weights()
        
        torch.save(self.esm3_model.state_dict(), path)
        print(f"Merged model saved to: {path}")
        
        for lora_layer in self.lora_layers.values():
            lora_layer.unmerge_weights()
    
    def print_trainable_parameters(self):
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Parameter Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Trainable ratio: {100 * trainable_params / total_params:.2f}%")


def create_esm3_lora_model(
    esm3_model,
    target_modules: List[str] = None,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1
) -> ESM3LoRAWrapper:
    """
    Create ESM3 LoRA model
    
    Args:
        esm3_model: Original ESM3 model
        target_modules: List of target module names
        r: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout probability
    
    Returns:
        ESM3LoRAWrapper: Wrapped LoRA model
    """
    return ESM3LoRAWrapper(
        esm3_model=esm3_model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout
    )

