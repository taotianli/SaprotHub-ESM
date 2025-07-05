import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotClassificationModel(SaprotBaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)
        
    def initialize_metrics(self, stage):
        # For newer versions of torchmetrics, need to specify task type
        if self.num_labels == 2:
            task = "binary"
        else:
            task = "multiclass"
        
        return {f"{stage}_acc": torchmetrics.Accuracy(task=task, num_classes=self.num_labels)}

    def forward(self, inputs=None, coords=None, sequences=None, **kwargs):
        # Handle different input formats
        if inputs is None and sequences is not None:
            inputs = {"sequences": sequences}
        elif inputs is None:
            inputs = kwargs
        
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # Handle sequences from ESM3-compatible dataset
        sequences = inputs.get("sequences", None)
        if sequences is not None:
            # Process sequences using ESM3
            from esm.sdk.api import ESMProtein
            
            features = []
            for seq in sequences:
                protein = ESMProtein(sequence=seq)
                with torch.no_grad():
                    # Encode protein using ESM3 model
                    encoded_protein = self.model.encode(protein)
                    
                    # Extract sequence embeddings
                    if hasattr(encoded_protein, 'sequence'):
                        seq_repr = encoded_protein.sequence
                        if torch.is_tensor(seq_repr):
                            # Apply mean pooling if it's a sequence of embeddings
                            if seq_repr.dim() > 1:
                                features.append(seq_repr.mean(dim=0))
                            else:
                                features.append(seq_repr)
                        else:
                            # Convert to tensor - use float32 first then convert to model dtype
                            tensor_repr = torch.tensor(seq_repr, dtype=torch.float32)
                            if tensor_repr.dim() > 1:
                                features.append(tensor_repr.mean(dim=0))
                            else:
                                features.append(tensor_repr)
                    else:
                        # Fallback: use a default embedding size
                        features.append(torch.zeros(2560, dtype=torch.float32))  # ESM3 embedding size
            
            # Stack and prepare features
            if features:
                try:
                    # Ensure all features have the same size
                    target_size = features[0].shape[0] if features[0].dim() > 0 else 2560
                    normalized_features = []
                    
                    for feat in features:
                        if feat.dim() == 0:
                            # Scalar tensor
                            norm_feat = torch.zeros(target_size, dtype=torch.float32)
                            norm_feat[0] = feat
                        elif feat.shape[0] != target_size:
                            # Resize to target size
                            if feat.shape[0] > target_size:
                                norm_feat = feat[:target_size]
                            else:
                                norm_feat = torch.cat([feat, torch.zeros(target_size - feat.shape[0], dtype=torch.float32)])
                        else:
                            norm_feat = feat
                        normalized_features.append(norm_feat)
                    
                    stacked_features = torch.stack(normalized_features)
                    
                except Exception as e:
                    # Fallback if stacking fails
                    batch_size = len(features)
                    stacked_features = torch.zeros(batch_size, 2560, dtype=torch.float32)
            else:
                stacked_features = torch.zeros(1, 2560, dtype=torch.float32)
        
        else:
            # Legacy handling for pre-encoded data
            encoded_proteins = inputs.get("inputs", inputs)
            features = []
            for protein in encoded_proteins:
                seq_attr = getattr(protein, 'sequence', None)
                if seq_attr is not None:
                    if torch.is_tensor(seq_attr):
                        features.append(seq_attr.mean(dim=0) if seq_attr.dim() > 1 else seq_attr)
                    else:
                        tensor_repr = torch.tensor(seq_attr, dtype=torch.float32)
                        features.append(tensor_repr.mean(dim=0) if tensor_repr.dim() > 1 else tensor_repr)
                else:
                    features.append(torch.zeros(2560, dtype=torch.float32))
            
            if features:
                stacked_features = torch.stack(features)
            else:
                stacked_features = torch.zeros(1, 2560, dtype=torch.float32)
        
        # Move to correct device and ensure proper dtype
        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        stacked_features = stacked_features.to(device=device, dtype=model_dtype)
        
        # Create classification head if not exists
        if not hasattr(self, 'classification_head'):
            input_dim = stacked_features.shape[-1]
            self.classification_head = torch.nn.Linear(input_dim, self.num_labels)
            self.classification_head = self.classification_head.to(device=device, dtype=model_dtype)
        
        # Forward pass - let PyTorch Lightning handle mixed precision automatically
        logits = self.classification_head(stacked_features)
        
        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        
        # Compute loss - PyTorch Lightning handles mixed precision automatically
        loss = cross_entropy(logits, label)
        
        # Update metrics - convert logits to predictions for metric calculation
        # Convert to float32 for metric computation to avoid precision issues
        preds = torch.argmax(logits.float(), dim=-1)
        for metric in self.metrics[stage].values():
            metric.update(preds.detach(), label)

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        log_dict = self.get_log_dict("test")
        # log_dict["test_loss"] = torch.cat(self.all_gather(self.test_outputs), dim=-1).mean()
        log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.output_test_metrics(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")
        # log_dict["valid_loss"] = torch.cat(self.all_gather(self.valid_outputs), dim=-1).mean()
        log_dict["valid_loss"] = torch.mean(torch.stack(self.valid_outputs))

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_acc"], mode="max")

        self.plot_valid_metrics_curve(log_dict)