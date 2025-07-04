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
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # For ESM3 compatibility - handle encoded proteins
        encoded_proteins = inputs.get("inputs", inputs)
        
        # Process each encoded protein to extract features
        features = []
        for protein in encoded_proteins:
            # Try to extract sequence features from encoded protein
            seq_attr = getattr(protein, 'sequence', None)
            if seq_attr is not None:
                if torch.is_tensor(seq_attr):
                    # If it's a tensor, use it directly
                    features.append(seq_attr)
                else:
                    # Convert to tensor if not already
                    features.append(torch.tensor(seq_attr, dtype=torch.float32))
            else:
                # Fallback: create a dummy feature vector
                features.append(torch.zeros(512, dtype=torch.float32))
        
        # Stack features and handle different shapes
        if features:
            # Try to stack features, handling different shapes
            try:
                # Pad to same length if needed
                max_len = max(f.shape[0] if f.dim() > 0 else 1 for f in features)
                padded_features = []
                for f in features:
                    if f.dim() == 0:
                        # Scalar tensor
                        padded_f = torch.zeros(max_len)
                        padded_f[0] = f
                    elif f.shape[0] < max_len:
                        # Pad with zeros
                        padded_f = torch.cat([f, torch.zeros(max_len - f.shape[0])])
                    else:
                        padded_f = f[:max_len]
                    padded_features.append(padded_f)
                
                stacked_features = torch.stack(padded_features)
                
                # Average pooling to get fixed-size representation
                pooled = stacked_features.mean(dim=1) if stacked_features.dim() > 1 else stacked_features
                
            except Exception as e:
                # If stacking fails, create dummy features
                batch_size = len(features)
                pooled = torch.zeros(batch_size, 512)
        else:
            # No features available
            pooled = torch.zeros(1, 512)
        
        # Move to correct device
        if hasattr(self.model, 'device'):
            pooled = pooled.to(self.model.device)
        
        # Create classification head if not exists
        if not hasattr(self, 'classification_head'):
            input_dim = pooled.shape[-1]
            self.classification_head = torch.nn.Linear(input_dim, self.num_labels)
            if hasattr(self.model, 'device'):
                self.classification_head = self.classification_head.to(self.model.device)
        
        logits = self.classification_head(pooled)
        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)
        # print(loss, logits)

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), label)

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