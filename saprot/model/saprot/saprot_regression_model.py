import torch.distributed as dist
import torchmetrics
import torch

from ..model_interface import register_model
from .base import SaprotBaseModel


@register_model
class SaprotRegressionModel(SaprotBaseModel):
    def __init__(self, test_result_path: str = None, **kwargs):
        """
        Args:
            test_result_path: path to save test result
            **kwargs: other arguments for SaprotBaseModel
        """
        self.test_result_path = test_result_path
        super().__init__(task="regression", **kwargs)
    
    def initialize_metrics(self, stage):
        return {f"{stage}_loss": torchmetrics.MeanSquaredError(),
                f"{stage}_spearman": torchmetrics.SpearmanCorrCoef(),
                f"{stage}_R2": torchmetrics.R2Score(),
                f"{stage}_pearson": torchmetrics.PearsonCorrCoef()}
    
    def forward(self, inputs, structure_info=None):
        if structure_info:
            # To be implemented
            raise NotImplementedError

        # For ESM models
        if hasattr(self.model, "esm"):
            # If backbone is frozen, the embedding will be the average of all residues, else it will be the
            # embedding of the <cls> token.
            if self.freeze_backbone:
                repr = torch.stack(self.get_hidden_states_from_dict(inputs, reduction="mean"))
                x = self.model.classifier.dropout(repr)
                x = self.model.classifier.dense(x)
                x = torch.tanh(x)
                x = self.model.classifier.dropout(x)
                logits = self.model.classifier.out_proj(x).squeeze(dim=-1)

            else:
               logits = self.model(**inputs).logits.squeeze(dim=-1)
        
         # For ProtBERT
        elif hasattr(self.model, "bert"):
            # 检查输入的token IDs是否在有效范围内
            vocab_size = self.model.bert.embeddings.word_embeddings.num_embeddings
            input_ids = inputs["input_ids"]
            if torch.max(input_ids) >= vocab_size:
                # 将超出范围的ID替换为UNK token ID
                unk_id = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 0
                inputs["input_ids"] = torch.where(input_ids < vocab_size, input_ids, torch.tensor(unk_id).to(input_ids.device))
            repr = self.model.bert(**inputs).last_hidden_state[:, 0]
            logits = self.model.classifier(repr).squeeze(dim=-1)

        # print("Logits:", logits)
        # print("Inputs:", inputs)    

        return logits

    def loss_func(self, stage, outputs, labels):
        fitness = labels['labels'].to(outputs)
        loss = torch.nn.functional.mse_loss(outputs, fitness)
        
        # print("Outputs:", outputs)
        # print("Labels:", fitness)
        
        # Update metrics
        for metric in self.metrics[stage].values():
            # Training is on half precision, but metrics expect float to compute correctly.
            metric.set_dtype(torch.float32)
            metric.update(outputs.detach(), fitness)
            # print(metric.update(outputs.detach(), fitness))
            # print(metric.compute())

            
        if stage == "train":
            log_dict = {"train_loss": loss.item()}
            self.log_info(log_dict)
            
            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            from torchmetrics.utilities.distributed import gather_all_tensors
            
            preds = self.test_spearman.preds
            preds[-1] = preds[-1].unsqueeze(dim=0) if preds[-1].shape == () else preds[-1]
            preds = torch.cat(gather_all_tensors(torch.cat(preds, dim=0)))
            
            targets = self.test_spearman.target
            targets[-1] = targets[-1].unsqueeze(dim=0) if targets[-1].shape == () else targets[-1]
            targets = torch.cat(gather_all_tensors(torch.cat(targets, dim=0)))

            if dist.get_rank() == 0:
                with open(self.test_result_path, 'w') as w:
                    w.write("pred\ttarget\n")
                    for pred, target in zip(preds, targets):
                        w.write(f"{pred.item()}\t{target.item()}\n")
        
        log_dict = self.get_log_dict("test")

        # if dist.get_rank() == 0:
        #     print(log_dict)

        self.output_test_metrics(log_dict)

        self.log_info(log_dict)
        self.reset_metrics("test")

    def on_validation_epoch_end(self):
        log_dict = self.get_log_dict("valid")

        # if dist.get_rank() == 0:
        #     print(log_dict)
        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_loss"], mode="min")
        
        self.plot_valid_metrics_curve(log_dict)