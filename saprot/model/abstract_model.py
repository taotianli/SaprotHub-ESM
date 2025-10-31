import torch
import abc
import os
import copy

# 不再使用PyTorch Lightning
# import pytorch_lightning as pl
from utils.lr_scheduler import *
from utils.others import TimeCounter
# from torch import distributed as dist


class AbstractModel(torch.nn.Module):
    def __init__(self,
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,):
        """

        Args:
            lr_scheduler: Kwargs for lr_scheduler
            optimizer_kwargs: Kwargs for optimizer_kwargs
            save_path: Save trained model
            from_checkpoint: Load model from checkpoint
            load_prev_scheduler: Whether load previous scheduler from checkpoint
            load_strict: Whether load model strictly
            save_weights_only: Whether save only weights or also optimizer and lr_scheduler
            
        """
        super().__init__()
        self.initialize_model()
        
        self.metrics = {}
        for stage in ["train", "valid", "test"]:
            stage_metrics = self.initialize_metrics(stage)
            # Rigister metrics as attributes
            for metric_name, metric in stage_metrics.items():
                setattr(self, metric_name, metric)
                
            self.metrics[stage] = stage_metrics
        
        if lr_scheduler_kwargs is None:
            # Default lr_scheduler
            self.lr_scheduler_kwargs = {
                "class": "ConstantLRScheduler",
                "init_lr": 0,
            }
            # print("No lr_scheduler_kwargs provided. The default learning rate is 0.")

        else:
            self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        if optimizer_kwargs is None:
            # Default optimizer
            self.optimizer_kwargs = {
                "class": "AdamW",
                "betas": (0.9, 0.98),
                "weight_decay": 0.01,
            }
            # print("No optimizer_kwargs provided. The default optimizer is AdamW.")
        else:
            self.optimizer_kwargs = optimizer_kwargs
        self.init_optimizers()

        self.save_path = save_path
        self.save_weights_only = save_weights_only
        
        # temp_step is used for accumulating gradients
        self.temp_step = 0
        self.step = 0
        self.epoch = 0
        
        self.load_prev_scheduler = load_prev_scheduler
        self.from_checkpoint = from_checkpoint
        if from_checkpoint:
            self.load_checkpoint(from_checkpoint)

    @abc.abstractmethod
    def initialize_model(self) -> None:
        """
        All model initialization should be done here
        Note that the whole model must be named as "self.model" for model saving and loading
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward propagation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_metrics(self, stage: str) -> dict:
        """
        Initialize metrics for each stage
        Args:
            stage: "train", "valid" or "test"
        
        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self, stage: str, outputs, labels) -> torch.Tensor:
        """

        Args:
            stage: "train", "valid" or "test"
            outputs: model outputs for calculating loss
            labels: labels for calculating loss

        Returns:
            loss

        """
        raise NotImplementedError

    @staticmethod
    def load_weights(model, weights):
        model_dict = model.state_dict()

        unused_params = []
        missed_params = list(model_dict.keys())

        for k, v in weights.items():
            if k in model_dict.keys():
                model_dict[k] = v
                missed_params.remove(k)

            else:
                unused_params.append(k)

        if len(missed_params) > 0:
            print(f"\033[31mSome weights of {type(model).__name__} were not "
                  f"initialized from the model checkpoint: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31mSome weights of the model checkpoint were not used: {unused_params}\033[0m")

        model.load_state_dict(model_dict)

    # optimizer_step方法已移除，因为不再使用PyTorch Lightning
    # 优化器步骤现在由训练循环直接处理
        
    # For pytorch-lightning 1.9.5
    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer,
    #     optimizer_idx: int = 0,
    #     optimizer_closure=None,
    #     on_tpu: bool = False,
    #     using_native_amp: bool = False,
    #     using_lbfgs: bool = False,
    # ) -> None:
    #     super().optimizer_step(
    #         epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    #     )
    #     self.temp_step += 1
    #     if self.temp_step == self.trainer.accumulate_grad_batches:
    #         self.step += 1
    #         self.temp_step = 0

    def on_train_epoch_end(self):
        self.epoch += 1
    
    # training_step, validation_step, test_step方法已移除
    # 这些功能现在由纯PyTorch训练循环处理
    
    def on_train_start(self) -> None:
        # Load previous scheduler
        if getattr(self, "prev_schechuler", None) is not None:
            try:
                self.step = self.prev_schechuler["global_step"]
                self.epoch = self.prev_schechuler["epoch"]
                self.best_value = self.prev_schechuler["best_value"]
                self.lr_scheduler.load_state_dict(self.prev_schechuler["lr_scheduler"])
                print(f"Previous training global step: {self.step}")
                print(f"Previous training epoch: {self.epoch}")
                print(f"Previous best value: {self.best_value}")
                print(f"Previous lr_scheduler: {self.prev_schechuler['lr_scheduler']}")
                
                # 纯PyTorch版本：直接加载优化器状态
                if "optimizer" in self.prev_schechuler:
                    self.optimizer.load_state_dict(self.prev_schechuler["optimizer"])

            except Exception as e:
                print(e)
                raise Exception("Error in loading previous scheduler. Please set load_prev_scheduler=False")
    
    def on_validation_epoch_start(self) -> None:
        setattr(self, "valid_outputs", [])
    
    def on_test_epoch_start(self) -> None:
        setattr(self, "test_outputs", [])
            
    def load_checkpoint(self, from_checkpoint: str) -> None:
        """
        Args:
            from_checkpoint:  Path to checkpoint.
        """
        
        # If ``from_checkpoint`` is a directory, load the checkpoint in it
        if os.path.isdir(from_checkpoint):
            basename = os.path.basename(from_checkpoint)
            from_checkpoint = os.path.join(from_checkpoint, f"{basename}.pt")

        # 检查检查点文件是否存在
        if not os.path.exists(from_checkpoint):
            # print(f" 警告: 检查点文件不存在: {from_checkpoint}")
            # print("🔄 跳过检查点加载，使用随机初始化的模型进行训练")
            return

        try:
            # print(f"📂 正在加载检查点: {from_checkpoint}")
            state_dict = torch.load(from_checkpoint, map_location=self.device)
            
            if "model" not in state_dict:
                # print(f"检查点文件格式错误: 缺少'model'键")
                # print("🔄 跳过检查点加载，使用随机初始化的模型进行训练")
                return
                
            self.load_weights(self.model, state_dict["model"])
            # print(f"检查点加载成功")
            
            if self.load_prev_scheduler:
                state_dict.pop("model")
                self.prev_schechuler = state_dict
                # print(f"调度器状态加载成功")
                
        except Exception as e:
            # print(f"加载检查点时出错: {str(e)}")
            # print("🔄 跳过检查点加载，使用随机初始化的模型进行训练")
            pass

    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        Save model to save_path
        Args:
            save_path: Path to save model
            save_info: Other info to save
            save_weights_only: Whether only save model weights
        """
        try:
            # 确保路径有.pt扩展名
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
                # print(f"添加.pt扩展名: {save_path}")
            
            # 确保目录路径存在
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                # print(f"📁 创建/确认保存目录: {dir_path}")
            
            # Test if directory is writable
            test_file = os.path.join(dir_path if dir_path else '.', '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                # print(f"目录可写: {dir_path if dir_path else '当前目录'}")
            except (OSError, IOError) as e:
                # If the original path is not writable, use a fallback path
                # print(f" 警告: 无法写入目录 {dir_path}, 使用备用路径")
                fallback_dir = os.path.join(os.getcwd(), 'model_checkpoints')
                os.makedirs(fallback_dir, exist_ok=True)
                filename = os.path.basename(save_path)
                if not filename.endswith('.pt'):
                    filename = filename + '.pt'
                save_path = os.path.join(fallback_dir, filename)
                # print(f"保存到备用路径: {save_path}")
            
            state_dict = {} if save_info is None else save_info
            state_dict["model"] = self.model.state_dict()
            
            # Convert model weights to fp32
            for k, v in state_dict["model"].items():
                state_dict["model"][k] = v.float()
                
            if not save_weights_only:
                state_dict["global_step"] = self.step
                state_dict["epoch"] = self.epoch
                state_dict["best_value"] = getattr(self, f"best_value", None)
                if hasattr(self, 'lr_scheduler'):
                    state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
                
                # 纯PyTorch版本：直接保存优化器状态
                if hasattr(self, 'optimizer'):
                    state_dict["optimizer"] = self.optimizer.state_dict()

            torch.save(state_dict, save_path)
            # print(f"模型检查点已保存到: {save_path}")
            
        except Exception as e:
            # print(f"保存检查点时出错: {e}")
            # Try to save to current directory as last resort
            try:
                fallback_path = os.path.join(os.getcwd(), 'emergency_checkpoint.pt')
                state_dict = {} if save_info is None else save_info
                state_dict["model"] = self.model.state_dict()
                torch.save(state_dict, fallback_path)
                # print(f"🚨 紧急检查点已保存到: {fallback_path}")
            except Exception as e2:
                # print(f"紧急保存也失败: {e2}")
                raise e

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        Check whether to save model. If save_path is not None and now_value is the best, save model.
        Args:
            now_value: Current metric value
            mode: "min" or "max", meaning whether the lower the better or the higher the better
            save_info: Other info to save
        """

        assert mode in ["min", "max"], "mode should be 'min' or 'max'"

        if self.save_path is not None:
            # In case there are variables to be included in the save path
            try:
                save_path = eval(f"f'{self.save_path}'")
            except:
                save_path = self.save_path
            
            # 确保路径有.pt扩展名
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            
            # print(f"检查保存条件，目标路径: {save_path}")
            
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
                # print(f"📁 创建保存目录: {dir_path}")
            
            # Check whether to save model
            best_value = getattr(self, f"best_value", None)
            if best_value is not None:
                if mode == "min" and now_value >= best_value or mode == "max" and now_value <= best_value:
                    # print(f"当前值 {now_value} 不是最佳值 (最佳: {best_value})，跳过保存")
                    return
                
            setattr(self, "best_value", now_value)
            # print(f"新的最佳值: {now_value}，准备保存模型")
                
            # 纯PyTorch版本：直接保存
            self.save_checkpoint(save_path, save_info, self.save_weights_only)
            
    def reset_metrics(self, stage) -> None:
        """
        Reset metrics for given stage
        Args:
            stage: "train", "valid" or "test"
        """
        for metric in self.metrics[stage].values():
            metric.reset()
    
    def get_log_dict(self, stage: str) -> dict:
        """
        Get log dict for the stage
        Args:
            stage: "train", "valid" or "test"

        Returns:
            A dictionary of metrics for the stage. Keys are metric names and values are metric values

        """
        log_dict = {}
        for name, metric in self.metrics[stage].items():
            try:
                log_dict[name] = metric.compute()
            except Exception as e:
                log_dict[name] = None
            
        return log_dict
    
    def log_info(self, info: dict) -> None:
        """
        Record metrics during training and testing
        Args:
            info: dict of metrics
        """
        # 纯PyTorch版本：简化日志记录
        if hasattr(self, "lr_scheduler"):
            info["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        info["epoch"] = self.epoch
        info["step"] = self.step
        # 可以在这里添加自定义的日志记录逻辑
        # print(f"[Step {self.step}] {info}")

    def init_optimizers(self):
        copy_optimizer_kwargs = copy.deepcopy(self.optimizer_kwargs)
        
        # No decay for layer norm and bias
        no_decay = ['LayerNorm.weight', 'bias']
        weight_decay = copy_optimizer_kwargs.pop("weight_decay")

        # Collect all trainable parameters from the entire model (including classification heads, etc.)
        # Use self.named_parameters() instead of self.model.named_parameters()
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]

        optimizer_cls = eval(f"torch.optim.{copy_optimizer_kwargs.pop('class')}")
        self.optimizer = optimizer_cls(optimizer_grouped_parameters,
                                       lr=self.lr_scheduler_kwargs['init_lr'],
                                       **copy_optimizer_kwargs)

        tmp_kwargs = copy.deepcopy(self.lr_scheduler_kwargs)
        lr_scheduler = tmp_kwargs.pop("class")
        self.lr_scheduler = eval(lr_scheduler)(self.optimizer, **tmp_kwargs)
    
    def configure_optimizers(self):
        return {"optimizer": self.optimizer,
                "lr_scheduler": {"scheduler": self.lr_scheduler,
                                 "interval": "step",
                                 "frequency": 1}
                }