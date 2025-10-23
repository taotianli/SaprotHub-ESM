"""
纯PyTorch抽象模型基类，不依赖PyTorch Lightning
用于替换原有的AbstractModel
"""

import torch
import torch.nn as nn
import abc
import os
import copy

from utils.lr_scheduler import *
from utils.others import TimeCounter


class AbstractModel(nn.Module):
    """不依赖PyTorch Lightning的抽象模型基类"""
    
    def __init__(self,
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,):
        """
        Args:
            lr_scheduler_kwargs: 学习率调度器参数
            optimizer_kwargs: 优化器参数
            save_path: 模型保存路径
            from_checkpoint: 从检查点加载模型
            load_prev_scheduler: 是否加载之前的调度器
            save_weights_only: 是否只保存权重
        """
        super().__init__()
        self.initialize_model()
        
        self.metrics = {}
        for stage in ["train", "valid", "test"]:
            stage_metrics = self.initialize_metrics(stage)
            # 注册指标为属性
            for metric_name, metric in stage_metrics.items():
                setattr(self, metric_name, metric)
            self.metrics[stage] = stage_metrics
        
        if lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {
                "class": "ConstantLRScheduler",
                "init_lr": 0,
            }
            print("未提供lr_scheduler_kwargs。默认学习率为0。")
        else:
            self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        if optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "class": "AdamW",
                "betas": (0.9, 0.98),
                "weight_decay": 0.01,
            }
            print("未提供optimizer_kwargs。默认优化器为AdamW。")
        else:
            self.optimizer_kwargs = optimizer_kwargs
        
        self.init_optimizers()

        self.save_path = save_path
        self.save_weights_only = save_weights_only
        
        # 步数和epoch计数
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
        所有模型初始化应在此完成
        注意整个模型必须命名为"self.model"以便模型保存和加载
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """前向传播"""
        raise NotImplementedError
    
    @abc.abstractmethod
    def initialize_metrics(self, stage: str) -> dict:
        """
        初始化每个阶段的指标
        Args:
            stage: "train", "valid" 或 "test"
        Returns:
            该阶段的指标字典。键为指标名称，值为指标对象
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self, stage: str, outputs, labels) -> torch.Tensor:
        """
        计算损失
        Args:
            stage: "train", "valid" 或 "test"
            outputs: 模型输出
            labels: 标签
        Returns:
            损失值
        """
        raise NotImplementedError

    @staticmethod
    def load_weights(model, weights):
        """加载权重"""
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
            print(f"\033[31m{type(model).__name__}的某些权重未从模型检查点初始化: {missed_params}\033[0m")

        if len(unused_params) > 0:
            print(f"\033[31m模型检查点的某些权重未被使用: {unused_params}\033[0m")

        model.load_state_dict(model_dict)

    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        self.epoch += 1
    
    def on_train_start(self):
        """训练开始时的回调"""
        # 加载之前的调度器
        if getattr(self, "prev_schechuler", None) is not None:
            try:
                self.step = self.prev_schechuler["global_step"]
                self.epoch = self.prev_schechuler["epoch"]
                self.best_value = self.prev_schechuler["best_value"]
                self.lr_scheduler.load_state_dict(self.prev_schechuler["lr_scheduler"])
                print(f"之前的训练全局步数: {self.step}")
                print(f"之前的训练epoch: {self.epoch}")
                print(f"之前的最佳值: {self.best_value}")
                print(f"之前的lr_scheduler: {self.prev_schechuler['lr_scheduler']}")
                
                # 加载优化器状态
                self.optimizer.load_state_dict(self.prev_schechuler["optimizer"])
            except Exception as e:
                print(e)
                raise Exception("加载之前的调度器时出错。请设置load_prev_scheduler=False")
    
    def on_validation_epoch_start(self) -> None:
        """验证epoch开始时的回调"""
        setattr(self, "valid_outputs", [])
    
    def on_test_epoch_start(self) -> None:
        """测试epoch开始时的回调"""
        setattr(self, "test_outputs", [])
            
    def load_checkpoint(self, from_checkpoint: str) -> None:
        """
        从检查点加载模型
        Args:
            from_checkpoint: 检查点路径
        """
        # 如果是目录，加载其中的检查点
        if os.path.isdir(from_checkpoint):
            basename = os.path.basename(from_checkpoint)
            from_checkpoint = os.path.join(from_checkpoint, f"{basename}.pt")

        # 检查检查点文件是否存在
        if not os.path.exists(from_checkpoint):
            return

        try:
            state_dict = torch.load(from_checkpoint, map_location='cpu')
            
            if "model" not in state_dict:
                return
                
            self.load_weights(self.model, state_dict["model"])
            
            if self.load_prev_scheduler:
                state_dict.pop("model")
                self.prev_schechuler = state_dict
                
        except Exception as e:
            pass

    def save_checkpoint(self, save_path: str, save_info: dict = None, save_weights_only: bool = True) -> None:
        """
        保存模型到save_path
        Args:
            save_path: 保存路径
            save_info: 其他要保存的信息
            save_weights_only: 是否只保存模型权重
        """
        try:
            # 确保路径有.pt扩展名
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            
            # 确保目录路径存在
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 测试目录是否可写
            test_file = os.path.join(dir_path if dir_path else '.', '.write_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (OSError, IOError) as e:
                # 如果原路径不可写，使用备用路径
                fallback_dir = os.path.join(os.getcwd(), 'model_checkpoints')
                os.makedirs(fallback_dir, exist_ok=True)
                filename = os.path.basename(save_path)
                if not filename.endswith('.pt'):
                    filename = filename + '.pt'
                save_path = os.path.join(fallback_dir, filename)
            
            state_dict = {} if save_info is None else save_info
            state_dict["model"] = self.model.state_dict()
            
            # 将模型权重转换为fp32
            for k, v in state_dict["model"].items():
                state_dict["model"][k] = v.float()
                
            if not save_weights_only:
                state_dict["global_step"] = self.step
                state_dict["epoch"] = self.epoch
                state_dict["best_value"] = getattr(self, f"best_value", None)
                state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
                state_dict["optimizer"] = self.optimizer.state_dict()

            torch.save(state_dict, save_path)
            
        except Exception as e:
            # 尝试保存到当前目录作为最后手段
            try:
                fallback_path = os.path.join(os.getcwd(), 'emergency_checkpoint.pt')
                state_dict = {} if save_info is None else save_info
                state_dict["model"] = self.model.state_dict()
                torch.save(state_dict, fallback_path)
            except Exception as e2:
                raise e

    def check_save_condition(self, now_value: float, mode: str, save_info: dict = None) -> None:
        """
        检查是否保存模型。如果save_path不为None且now_value是最佳值，则保存模型。
        Args:
            now_value: 当前指标值
            mode: "min" 或 "max"，表示越低越好还是越高越好
            save_info: 其他要保存的信息
        """
        assert mode in ["min", "max"], "mode应为'min'或'max'"

        if self.save_path is not None:
            # 以防保存路径中有变量
            try:
                save_path = eval(f"f'{self.save_path}'")
            except:
                save_path = self.save_path
            
            # 确保路径有.pt扩展名
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            
            dir_path = os.path.dirname(save_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 检查是否保存模型
            best_value = getattr(self, f"best_value", None)
            if best_value is not None:
                if mode == "min" and now_value >= best_value or mode == "max" and now_value <= best_value:
                    return
                
            setattr(self, "best_value", now_value)
            self.save_checkpoint(save_path, save_info, self.save_weights_only)
            
    def reset_metrics(self, stage) -> None:
        """
        重置给定阶段的指标
        Args:
            stage: "train", "valid" 或 "test"
        """
        for metric in self.metrics[stage].values():
            metric.reset()
    
    def get_log_dict(self, stage: str) -> dict:
        """
        获取该阶段的日志字典
        Args:
            stage: "train", "valid" 或 "test"
        Returns:
            该阶段的指标字典。键为指标名称，值为指标值
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
        在训练和测试期间记录指标
        Args:
            info: 指标字典
        """
        # 纯PyTorch版本不需要logger，直接打印
        if hasattr(self, "lr_scheduler"):
            info["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        info["epoch"] = self.epoch
        # print(f"Step {self.step}: {info}")

    def init_optimizers(self):
        """初始化优化器和学习率调度器"""
        copy_optimizer_kwargs = copy.deepcopy(self.optimizer_kwargs)
        
        # 层归一化和偏置不进行权重衰减
        no_decay = ['LayerNorm.weight', 'bias']
        weight_decay = copy_optimizer_kwargs.pop("weight_decay")

        # 收集所有可训练参数
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
        """返回优化器和学习率调度器配置"""
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

