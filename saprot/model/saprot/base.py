import torch
import os

from typing import List, Dict
from data.pdb2feature import batch_coords2feature
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmTokenizer,
    BertTokenizer,
)
from easydict import EasyDict
from ..abstract_model import AbstractModel

import matplotlib.pyplot as plt


class SaprotBaseModel(AbstractModel):
    """
    Base class for SaProt models using ESM3
    """
    
    def __init__(self,
                 task: str,
                 config_path: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 gradient_checkpointing: bool = False,
                 lora_kwargs: dict = None,  # 保留参数但不使用
                 **kwargs):
        super().__init__(**kwargs)
        
        self.task = task
        self.config_path = config_path
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs  # 保留但不使用
        
        # Initialize model
        self.initialize_model()
        
        # Initialize metrics
        self.initialize_metrics_dict()
        
        # Initialize optimizers
        self.init_optimizers()
    
    def initialize_model(self):
        """Initialize ESM3 model"""
        from esm.models.esm3 import ESM3

        # 从config_path确定ESM3模型名称
        if self.config_path and self.config_path != "esm3-open":
            # 如果提供了具体的config_path，使用它
            esm3_model_name = self.config_path
            print(f"🔧 从指定路径加载ESM3模型: {esm3_model_name}")
        else:
            # 默认使用esm3-open
            esm3_model_name = "esm3-open"
            print(f"🔧 使用默认ESM3模型: {esm3_model_name}")

        print(f"🚀 开始加载ESM3模型...")
        self.model = ESM3.from_pretrained(esm3_model_name)
        print(f"✅ ESM3模型加载完成: {esm3_model_name}")

        # 打印模型信息
        if hasattr(self.model, 'config'):
            print(f"📊 模型配置信息: {self.model.config}")

        print(f"🎯 模型设备: {next(self.model.parameters()).device}")
        print(f"🎯 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        if self.extra_config is None:
            self.extra_config = {}

        # 冻结骨干网络（如果需要）
        if self.freeze_backbone:
            print(f"❄️ 冻结ESM3骨干网络参数...")
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"❄️ 骨干网络已冻结")

        # 启用梯度检查点（如果需要）
        if self.gradient_checkpointing:
            print(f"🔄 启用梯度检查点...")
            self.model.gradient_checkpointing_enable()
            print(f"✅ 梯度检查点已启用")
    
    def initialize_metrics(self, stage: str) -> dict:
        return {}
    
    def get_hidden_states_from_dict(self, inputs: dict, reduction: str = None) -> list:
        """
        Get hidden representations from input dict - using ESM3 encoding

        Args:
            inputs:  A dictionary of inputs containing ESM3 encoded data.
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        from esm.sdk.api import ESMProtein
        
        # Get encoded proteins from inputs
        encoded_proteins = inputs.get("inputs", inputs)
        
        repr_list = []
        for protein in encoded_proteins:
            # Use ESM3 model to get hidden states
            with torch.no_grad():
                # Get embeddings from the ESM3 model
                output = self.model.forward(protein)
                
                # Extract sequence embeddings
                if hasattr(output, 'sequence'):
                    hidden_states = output.sequence
                elif hasattr(output, 'embeddings'):
                    hidden_states = output.embeddings
                else:
                    # Fallback: try to get any tensor attribute
                    hidden_states = None
                    for attr_name in dir(output):
                        attr = getattr(output, attr_name)
                        if torch.is_tensor(attr) and attr.dim() >= 2:
                            hidden_states = attr
                            break
                    
                    if hidden_states is None:
                        # Final fallback
                        hidden_states = torch.zeros(512, 1024)  # Default size
                
                # Apply reduction if specified
                if reduction == "mean":
                    if hidden_states.dim() > 1:
                        repr = hidden_states.mean(dim=0)
                    else:
                        repr = hidden_states
                else:
                    repr = hidden_states
                
                repr_list.append(repr)
        
        return repr_list

    
    def get_hidden_states_from_seqs(self, seqs: list, reduction: str = None) -> list:
        """
        Get hidden representations of protein sequences - modified for ESM3 compatibility

        Args:
            seqs: A list of protein sequences
            reduction: Whether to reduce the hidden states. If None, the hidden states are not reduced. If "mean",
                        the hidden states are averaged over the sequence length.

        Returns:
            hidden_states: A list of tensors. Each tensor is of shape [L, D], where L is the sequence length and D is
                            the hidden dimension.
        """
        # Use ESM3 encoding for sequences
        from esm.sdk.api import ESMProtein
        
        repr_list = []
        device = self.model.device if hasattr(self.model, 'device') else 'cpu'
        
        for seq in seqs:
            protein = ESMProtein(sequence=seq)
            with torch.no_grad():
                encoded_protein = self.model.encode(protein)
                # Extract sequence representation
                seq_attr = getattr(encoded_protein, 'sequence', None)
                if seq_attr is not None:
                    if reduction == "mean":
                        repr = seq_attr.mean(dim=0) if torch.is_tensor(seq_attr) else torch.tensor(seq_attr).mean(dim=0)
                    else:
                        repr = seq_attr if torch.is_tensor(seq_attr) else torch.tensor(seq_attr)
                else:
                    # Fallback
                    repr = torch.zeros(512)
                
                repr_list.append(repr.to(device))
        
        return repr_list
    
    def add_bias_feature(self, inputs, coords: List[Dict]) -> torch.Tensor:
        """
        Add structure information as biases to attention map. This function is used to add structure information
        to the model as Evoformer does.

        Args:
            inputs: A dictionary of inputs. It should contain keys ["input_ids", "attention_mask", "token_type_ids"].
            coords: Coordinates of backbone atoms. Each element is a dictionary with keys ["N", "CA", "C", "O"].

        Returns
            pair_feature: A tensor of shape [B, L, L, 407]. Here 407 is the RBF of distance(400) + angle(7).
        """
        inputs["pair_feature"] = batch_coords2feature(coords, self.model.device)
        return inputs
    
    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        Save model checkpoint with proper directory creation
        """
        try:
            # Create directory if it doesn't exist
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                # print(f"📁 创建保存目录: {dir_path}")
            
            # Call parent save_checkpoint method
            super().save_checkpoint(save_path, save_info, save_weights_only)
            print(f"💾 模型checkpoint已保存到: {save_path}")
            
        except Exception as e:
            print(f"❌ 保存checkpoint失败: {str(e)}")
            # Try to save to current directory as fallback
            try:
                fallback_path = os.path.join(os.getcwd(), 'model_checkpoint.pt')
                super().save_checkpoint(fallback_path, save_info, save_weights_only)
                print(f"💾 fallback checkpoint已保存到: {fallback_path}")
            except Exception as e2:
                print(f"❌ fallback保存也失败: {str(e2)}")
                raise e
    
    def output_test_metrics(self, log_dict):
        # Remove valid_loss from log_dict when the task is classification
        if "test_acc" in log_dict:
            log_dict.pop("test_loss")
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and self.num_labels > 2:
            log_dict.pop("test_mcc")
        
        METRIC_MAP = {
            "test_acc": "Classification accuracy (Acc)",
            "test_loss": "Root mean squared error (RMSE)",  # Only for regression task
            "test_mcc": "Matthews correlation coefficient (MCC)",
            "test_r2": "Coefficient of determination (R^2)",
            "test_spearman": "Spearman correlation",
            "test_pearson": "Pearson correlation",
        }
        
        print('=' * 100)
        print('Evaluation results on the test set:')
        flag = False
        for key, value in log_dict.items():
            if value is not None:
                print_value = value.item()
            else:
                print_value = torch.nan
                flag = True
            
            print(f"{METRIC_MAP[key.lower()]}: {print_value}")
        
        if "classification" not in self.task and flag:
            print("\033[31m\nWarning: To calculate some metrics (R^2, Spearman correlation, Pearson correlation), "
                  "a minimum of two examples from the validation/test set is required.\033[0m")
        print('=' * 100)
    
    def plot_valid_metrics_curve(self, log_dict):
        if not hasattr(self, 'grid'):
            from google.colab import widgets
            width = 400 * len(log_dict)
            height = 400
            self.grid = widgets.Grid(1, 1, header_row=False, header_column=False,
                                     style=f'width:{width}px; height:{height}px')
        
        # Remove valid_loss from log_dict when the task is classification
        if "valid_acc" in log_dict:
            log_dict.pop("valid_loss")
        
        # Remove mcc metric if the number of classes is greater than 2
        if self.task == "token_classification" and self.num_labels > 2:
            log_dict.pop("valid_mcc")
        
        METRIC_MAP = {
            "valid_acc": "Classification accuracy (Acc)",
            "valid_loss": "Root mean squared error (RMSE)",  # Only for regression task
            "valid_mcc": "Matthews correlation coefficient (MCC)",
            "valid_r2": "Coefficient of determination (R$^2$)",
            "valid_spearman": "Spearman correlation",
            "valid_pearson": "Pearson correlation",
        }
        
        with self.grid.output_to(0, 0):
            self.grid.clear_cell()
            
            fig = plt.figure(figsize=(6 * len(log_dict), 6))
            ax = []
            self.valid_metrics_list['step'].append(int(self.step))
            for idx, metric in enumerate(log_dict.keys()):
                value = torch.nan if log_dict[metric] is None else log_dict[metric].detach().cpu().item()
                
                if metric in self.valid_metrics_list:
                    self.valid_metrics_list[metric].append(value)
                else:
                    self.valid_metrics_list[metric] = [value]
                
                ax.append(fig.add_subplot(1, len(log_dict), idx + 1))
                ax[idx].set_title(METRIC_MAP[metric.lower()])
                ax[idx].set_xlabel('step')
                ax[idx].set_ylabel(METRIC_MAP[metric.lower()])
                ax[idx].plot(self.valid_metrics_list['step'], self.valid_metrics_list[metric], marker='o')
            
            import ipywidgets
            import markdown
            from IPython.display import display
            
            hint = ipywidgets.HTML(
                markdown.markdown(
                    f"### The model is saved to {self.save_path}.\n\n"
                    "### Evaluation results on the validation set are shown below.\n\n"
                    "### You can check <a href='https://github.com/westlake-repl/SaprotHub/wiki/SaprotHub-v2-(latest)#3-how-can-i-monitor-model-performance-during-training-and-detect-overfitting' target='blank'>here</a> to see how to judge the overfitting of your model."
                )
            )
            display(hint)
            # plt.tight_layout()
            plt.show()
