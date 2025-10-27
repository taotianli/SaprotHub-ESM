import torch
import os

from typing import List, Dict
from data.pdb2feature import batch_coords2feature

# transformers库对ESM3完全不需要！
# ESM3使用自己的模型架构和编码方式，不依赖HuggingFace的transformers
# 原来的代码已经全部注释掉，这些导入完全没用
# from transformers import (
#     AutoConfig,
#     AutoTokenizer,
#     AutoModelForMaskedLM,
#     AutoModelForSequenceClassification,
#     AutoModelForTokenClassification,
#     EsmForMaskedLM,
#     EsmForSequenceClassification,
#     EsmTokenizer,
#     BertTokenizer,
# )

from easydict import EasyDict
from ..abstract_model import AbstractModel

import matplotlib.pyplot as plt


class SaprotBaseModel(AbstractModel):
    """
    ESM base model. It cannot be used directly but provides model initialization for downstream tasks.
    """
    
    def __init__(self,
                 task: str,
                 config_path: str,
                 extra_config: dict = None,
                 load_pretrained: bool = False,
                 freeze_backbone: bool = False,
                 gradient_checkpointing: bool = False,
                 lora_kwargs: dict = None,
                 **kwargs):
        """
        Args:
            task: Task name。

            config_path: Path to the config file of huggingface esm model

            extra_config: Extra config for the model

            load_pretrained: Whether to load pretrained weights of base model

            freeze_backbone: Whether to freeze the backbone of the model

            gradient_checkpointing: Whether to enable gradient checkpointing

            lora_kwargs: LoRA configuration

            **kwargs: Other arguments for AbstractModel
        """
        assert task in ['classification', 'token_classification', 'regression', 'lm', 'base']
        self.task = task
        self.config_path = config_path
        self.extra_config = extra_config
        self.load_pretrained = load_pretrained
        self.freeze_backbone = freeze_backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs
        super().__init__(**kwargs)
        
        # After all initialization done, lora technique is applied if needed
        if self.lora_kwargs is not None:
            # No need to freeze backbone if LoRA is used
            self.freeze_backbone = False
            
            self.lora_kwargs = EasyDict(lora_kwargs)
            self._init_lora()
        
        self.valid_metrics_list = {}
        self.valid_metrics_list['step'] = []
    
    def _init_lora(self):
        # Check if this is an ESM3 or ESMC model
        try:
            from esm.models.esm3 import ESM3
            is_esm3 = isinstance(self.model, ESM3)
        except:
            is_esm3 = False
        
        try:
            from esm.models.esmc import ESMC
            is_esmc = isinstance(self.model, ESMC)
        except:
            is_esmc = False
        
        if is_esm3 or is_esmc:
            # Use ESM3/ESMC LoRA
            model_name = "ESMC" if is_esmc else "ESM3"
            print(f"🔧 Using {model_name} LoRA...")
            # 使用绝对导入，因为saprot包已安装
            from saprot.utils.esm3_lora import create_esm3_lora_model
            
            # Get LoRA configuration
            r = getattr(self.lora_kwargs, "r", 16)
            lora_alpha = getattr(self.lora_kwargs, "lora_alpha", 32.0)
            lora_dropout = getattr(self.lora_kwargs, "lora_dropout", 0.1)
            target_modules = getattr(self.lora_kwargs, "target_modules", None)
            
            # Wrap model with LoRA
            self.model = create_esm3_lora_model(
                esm3_model=self.model,
                target_modules=target_modules,
                r=r,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            
            print(f"✅ {model_name} LoRA initialized successfully")
            self.model.print_trainable_parameters()
            
            # After LoRA model is initialized, add trainable parameters to optimizer
            self.init_optimizers()
            return
        
        # Original PEFT-based LoRA for non-ESM3 models
        from peft import (
            LoraConfig,
            # PeftModelForSequenceClassification,
            # get_peft_model
        )
        
        from .self_peft.mapping import get_peft_model
        from .self_peft.peft_model import PeftModelForSequenceClassification
        
        is_trainable = getattr(self.lora_kwargs, "is_trainable", False)
        config_list = getattr(self.lora_kwargs, "config_list", [])
        assert self.lora_kwargs.num_lora >= len(config_list), ("The number of LoRA models should be greater than or "
                                                               "equal to the number of weight files.")
        for i in range(self.lora_kwargs.num_lora):
            adapter_name = f"adapter_{i}" if self.lora_kwargs.num_lora > 1 else "default"
            
            # Load pre-trained LoRA weights
            if i < len(config_list):
                lora_config_path = config_list[i].lora_config_path
                if i == 0:
                    # If i == 0, initialize a PEFT model
                    self.model = PeftModelForSequenceClassification.from_pretrained(self.model,
                                                                                    lora_config_path,
                                                                                    adapter_name=adapter_name,
                                                                                    is_trainable=is_trainable)
                else:
                    self.model.load_adapter(lora_config_path, adapter_name=adapter_name, is_trainable=is_trainable)
            
            # Initialize LoRA model for training
            else:
                lora_config = {
                    "task_type": "SEQ_CLS",
                    "target_modules": ["query", "key", "value", "intermediate.dense", "output.dense"],
                    "modules_to_save": ["classifier"],
                    "inference_mode": False,
                    "r": getattr(self.lora_kwargs, "r", 8),
                    "lora_dropout": getattr(self.lora_kwargs, "lora_dropout", 0.0),
                    "lora_alpha": getattr(self.lora_kwargs, "lora_alpha", 16),
                }
                
                lora_config = LoraConfig(**lora_config)
                
                if i == 0:
                    # If i == 0, initialize a PEFT model
                    self.model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
                
                else:
                    self.model.add_adapter(adapter_name, lora_config)
        
        if self.lora_kwargs.num_lora > 1:
            # Multiple LoRA models only support inference mode
            print("Multiple LoRA models are used. This only supports inference mode. If you want to train the model,"
                  "set num_lora to 1.")
            
            # Replace the normal forward function with the lora ensemble function, which averages the outputs of all
            # LoRA models.
            def lora_forward(func):
                
                def forward(*args, **kwargs):
                    logits_list = []
                    ori_shape = None
                    
                    for i in range(self.lora_kwargs.num_lora):
                        adapter_name = f"adapter_{i}"
                        self.model.set_adapter(adapter_name)
                        logits = func(*args, **kwargs)
                        logits_list.append(logits)
                        
                        if ori_shape is None:
                            ori_shape = logits.shape
                    
                    logits = torch.stack(logits_list, dim=0)
                    
                    # For classification task, final labels are voted by all LoRA models
                    if len(ori_shape) == 2:
                        logits = logits.permute(1, 0, 2)
                        preds = logits.argmax(dim=-1)
                        preds = torch.mode(preds, dim=1).values
                        
                        # Generate dummy logits to match the original output
                        dummy_logits = torch.zeros(ori_shape).to(logits)
                        for i, pred in enumerate(preds):
                            dummy_logits[i, pred] = 1.0
                    
                    # For regression task, final labels are averaged among all LoRA models
                    else:
                        dummy_logits = logits.mean(dim=0)
                    
                    return dummy_logits.detach()
                
                return forward
            
            self.forward = lora_forward(self.forward)
        
        print(f"Now active LoRA model: {self.model.active_adapter}")
        self.model.print_trainable_parameters()
        
        # After LoRA model is initialized, add trainable parameters to optimizer)
        self.init_optimizers()
    
    def initialize_model(self):
        # Initialize tokenizer - commented out for ESM3 compatibility
        # self.tokenizer = AutoTokenizer.from_pretrained(self.config_path)
        
        # Initialize different models according to task - commented out for ESM3 compatibility
        # config = AutoConfig.from_pretrained(self.config_path)
        # if self.extra_config:
        #     for k, v in self.extra_config.items():
        #         setattr(config, k, v)
        # 
        # else:
        #     self.extra_config = {}
        # 
        # if self.task == 'classification':
        #     # Note that self.num_labels should be set in child classes
        #     if self.load_pretrained:
        #         self.model = AutoModelForSequenceClassification.from_pretrained(
        #             self.config_path, num_labels=self.num_labels, **self.extra_config)
        #     
        #     else:
        #         config.num_labels = self.num_labels
        #         self.model = AutoModelForSequenceClassification.from_config(config)
        # 
        # if self.task == 'token_classification':
        #     # Note that self.num_labels should be set in child classes
        #     if self.load_pretrained:
        #         self.model = AutoModelForTokenClassification.from_pretrained(
        #             self.config_path, num_labels=self.num_labels, **self.extra_config)
        #     
        #     else:
        #         config.num_labels = self.num_labels
        #         self.model = AutoModelForTokenClassification.from_config(config)
        # 
        # elif self.task == 'regression':
        #     if self.load_pretrained:
        #         self.model = AutoModelForSequenceClassification.from_pretrained(
        #             self.config_path, num_labels=1, **self.extra_config)
        #     
        #     else:
        #         config.num_labels = 1
        #         self.model = AutoModelForSequenceClassification.from_config(config)
        # 
        # elif self.task == 'lm':
        #     if self.load_pretrained:
        #         self.model = AutoModelForMaskedLM.from_pretrained(self.config_path, **self.extra_config)
        #     
        #     else:
        #         self.model = AutoModelForMaskedLM.from_config(config)
        # 
        # elif self.task == 'base':
        #     if self.load_pretrained:
        #         self.model = AutoModelForMaskedLM.from_pretrained(self.config_path, **self.extra_config)
        #     
        #     else:
        #         self.model = AutoModelForMaskedLM.from_config(config)
        #     
        #     if isinstance(self.model, EsmForMaskedLM) or isinstance(self.model, EsmForSequenceClassification):
        #         self.model.lm_head = None
        # 
        # if isinstance(self.model, EsmForMaskedLM) or isinstance(self.model, EsmForSequenceClassification):
        #     # Remove contact head
        #     self.model.esm.contact_head = None
        #     
        #     # Remove position embedding if the embedding type is ``rotary``
        #     if config.position_embedding_type == "rotary":
        #         self.model.esm.embeddings.position_embeddings = None
        #     
        #     # Set gradient checkpointing
        #     self.model.esm.encoder.gradient_checkpointing = self.gradient_checkpointing
        # 
        # # Freeze the backbone of the model
        # if self.freeze_backbone:
        #     for param in self.model.esm.parameters():
        #         param.requires_grad = False
        
        # Initialize ESM3 or ESMC model based on config_path
        # 判断是使用ESM3还是ESMC
        if self.config_path and "esmc" in self.config_path.lower():
            # 使用ESMC模型
            from esm.models.esmc import ESMC
            
            esmc_model_name = self.config_path if self.config_path else "esmc_300m"
            # print(f"🔧 从指定路径加载ESMC模型: {esmc_model_name}")
            # print(f"🚀 开始加载ESMC模型...")
            self.model = ESMC.from_pretrained(esmc_model_name)
            # print(f"✅ ESMC模型加载完成: {esmc_model_name}")
            self.model_type = "esmc"
        else:
            # 使用ESM3模型
            from esm.models.esm3 import ESM3

            # 从config_path确定ESM3模型名称
            if self.config_path and self.config_path != "esm3-open":
                # 如果提供了具体的config_path，使用它
                esm3_model_name = self.config_path
                # print(f"🔧 从指定路径加载ESM3模型: {esm3_model_name}")
            else:
                # 默认使用esm3-open
                esm3_model_name = "esm3-open"
                # print(f"🔧 使用默认ESM3模型: {esm3_model_name}")

            # print(f"🚀 开始加载ESM3模型...")
            self.model = ESM3.from_pretrained(esm3_model_name)
            # print(f"✅ ESM3模型加载完成: {esm3_model_name}")
            self.model_type = "esm3"

        # 打印模型信息
        # if hasattr(self.model, 'config'):
        #     print(f"📊 模型配置信息: {self.model.config}")

        # print(f"🎯 模型设备: {next(self.model.parameters()).device}")
        # print(f"🎯 模型数据类型: {next(self.model.parameters()).dtype}")
        # print(f"🎯 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")

        if self.extra_config is None:
            self.extra_config = {}

        # 冻结骨干网络（如果需要）
        if self.freeze_backbone:
            print(f"❄️ 冻结ESM3骨干网络参数...")
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"❄️ 骨干网络已冻结")

        # # Disable the pooling layer
        # backbone = getattr(self.model, "esm", self.model.bert)
        # backbone.pooler = None
    
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
            # print(f"💾 模型checkpoint已保存到: {save_path}")
            
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
        
        # print('=' * 100)
        # print('Evaluation results on the test set:')
        # flag = False
        # for key, value in log_dict.items():
        #     if value is not None:
        #         if isinstance(value, torch.Tensor):
        #             print_value = value.item()
        #         else:
        #             print_value = float(value)
        #     else:
        #         print_value = torch.nan
        #         flag = True
        #     
        #     print(f"{METRIC_MAP[key.lower()]}: {print_value}")
        # 
        # if "classification" not in self.task and flag:
        #     print("\033[31m\nWarning: To calculate some metrics (R^2, Spearman correlation, Pearson correlation), "
        #           "a minimum of two examples from the validation/test set is required.\033[0m")
        # print('=' * 100)
    
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
        
        # Filter out keys that are not valid metrics (e.g., learning_rate, epoch)
        metrics_to_plot = {k: v for k, v in log_dict.items() if k.lower() in METRIC_MAP}
        
        with self.grid.output_to(0, 0):
            self.grid.clear_cell()
            
            fig = plt.figure(figsize=(6 * len(metrics_to_plot), 6))
            ax = []
            self.valid_metrics_list['step'].append(int(self.step))
            for idx, metric in enumerate(metrics_to_plot.keys()):
                if metrics_to_plot[metric] is None:
                    value = torch.nan
                elif isinstance(metrics_to_plot[metric], torch.Tensor):
                    value = metrics_to_plot[metric].detach().cpu().item()
                else:
                    value = float(metrics_to_plot[metric])
                
                if metric in self.valid_metrics_list:
                    self.valid_metrics_list[metric].append(value)
                else:
                    self.valid_metrics_list[metric] = [value]
                
                ax.append(fig.add_subplot(1, len(metrics_to_plot), idx + 1))
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
            
            # Print accuracy values for debugging
            # print("\n" + "=" * 100)
            # print("Validation metrics at each step:")
            # print("=" * 100)
            # for metric in metrics_to_plot.keys():
            #     if metric in self.valid_metrics_list:
            #         print(f"\n{METRIC_MAP[metric.lower()]}:")
            #         for step, val in zip(self.valid_metrics_list['step'], self.valid_metrics_list[metric]):
            #             print(f"  Step {step}: {val:.6f}")
            # print("=" * 100)
