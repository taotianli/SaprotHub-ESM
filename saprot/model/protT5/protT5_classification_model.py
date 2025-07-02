import torchmetrics
import torch
import torch.distributed as dist

from torch.nn.functional import cross_entropy
from ..model_interface import register_model
from .base import ProtT5BaseModel


@register_model
class ProtT5ClassificationModel(ProtT5BaseModel):
    def __init__(self, num_labels: int, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for SaprotBaseModel
        """
        self.num_labels = num_labels
        super().__init__(task="classification", **kwargs)

    def initialize_model(self):
        super().initialize_model()

        hidden_size = self.model.config.hidden_size
        classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.num_labels)
        )

        setattr(self.model, "classifier", classifier)
        
    def initialize_metrics(self, stage):
        return {f"{stage}_acc": torchmetrics.Accuracy()}

    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)
        
        # Get the input_ids and attention_mask
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # If backbone is frozen, the embedding will be the average of all residues
        if self.freeze_backbone:
            repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
            x = self.model.classifier.dropout(repr)
            x = self.model.classifier.dense(x)
            x = torch.tanh(x)
            x = self.model.classifier.dropout(x)
            logits = self.model.classifier.out_proj(x)

        else:
            # logits = self.model(**inputs).logits
            # print(input_ids.shape, attention_mask.shape)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            repr = outputs.last_hidden_state[:, 1: -1].mean(dim=1)
            logits = self.model.classifier(repr)
        
        return logits

    def loss_func(self, stage, logits, labels):
        label = labels['labels']
        loss = cross_entropy(logits, label)

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
        print('='*100)
        print('Test Result:')
        for key, value in log_dict.items():
            print(f"{key}: {value.item()}")
        print('='*100)
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