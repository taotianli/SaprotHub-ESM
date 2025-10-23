import os
import copy
# 不再使用 PyTorch Lightning
# import pytorch_lightning as pl
import datetime
import wandb

# 不再使用 PyTorch Lightning 的 loggers 和 strategies
# from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy, Strategy
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface

################################################################################
################################ load model ####################################
################################################################################
def my_load_model(config):
    model_config = copy.deepcopy(config)
    model_type = model_config.pop("model_py_path")

    if "kwargs" in model_config.keys():
        kwargs = model_config.pop('kwargs')
    else:
        kwargs = {}
    
    model_config.update(kwargs)

    if model_type == "saprot/saprot_classification_model":
      from model.saprot.saprot_classification_model import SaprotClassificationModel
      return SaprotClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_token_classification_model":
      from model.saprot.saprot_token_classification_model import SaprotTokenClassificationModel
      return SaprotTokenClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_regression_model":
      if 'num_labels' in model_config: del model_config['num_labels']
      from model.saprot.saprot_regression_model import SaprotRegressionModel
      return SaprotRegressionModel(**model_config)

    if model_type == "saprot/saprot_pair_classification_model":
      from model.saprot.saprot_pair_classification_model import SaprotPairClassificationModel
      return SaprotPairClassificationModel(**model_config)
    
    if model_type == "saprot/saprot_pair_regression_model":
      if 'num_labels' in model_config: del model_config['num_labels']
      from model.saprot.saprot_pair_regression_model import SaprotPairRegressionModel
      return SaprotPairRegressionModel(**model_config)
    
    if model_type == "protT5/protT5_classification_model":
      from model.protT5.protT5_classification_model import ProtT5ClassificationModel
      return ProtT5ClassificationModel(**model_config)
    
    if model_type == "protT5/protT5_regression_model":
      from model.protT5.protT5_regression_model import ProtT5RegressionModel
      return ProtT5RegressionModel(**model_config)
    
    if model_type == "protT5/protT5_token_classification_model":
      from model.protT5.protT5_token_classification_model import ProtT5TokenClassificationModel
      return ProtT5TokenClassificationModel(**model_config)


################################################################################
################################ load dataset ##################################
################################################################################
def my_load_dataset(config):
    dataset_config = copy.deepcopy(config)
    dataset_type = dataset_config.pop("dataset_py_path")
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)

    if dataset_type == "saprot/saprot_classification_dataset":
      from dataset.saprot.saprot_classification_dataset import SaprotClassificationDataset
      return SaprotClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_token_classification_dataset":
      if 'plddt_threshold' in dataset_config: del dataset_config['plddt_threshold']
      from dataset.saprot.saprot_token_classification_dataset import SaprotTokenClassificationDataset
      return SaprotTokenClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_regression_dataset":
      from dataset.saprot.saprot_regression_dataset import SaprotRegressionDataset
      return SaprotRegressionDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_pair_classification_dataset":
      from dataset.saprot.saprot_pair_classification_dataset import SaprotPairClassificationDataset
      return SaprotPairClassificationDataset(**dataset_config)
    
    if dataset_type == "saprot/saprot_pair_regression_dataset":
      from dataset.saprot.saprot_pair_regression_dataset import SaprotPairRegressionDataset
      return SaprotPairRegressionDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_classification_dataset":
      from dataset.protT5.protT5_classification_dataset import ProtT5ClassificationDataset
      return ProtT5ClassificationDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_regression_dataset":
      from dataset.protT5.protT5_regression_dataset import ProtT5RegressionDataset
      return ProtT5RegressionDataset(**dataset_config)
    
    if dataset_type == "protT5/protT5_token_classification_dataset":
      from dataset.protT5.protT5_token_classification_dataset import ProtT5TokenClassificationDataset
      return ProtT5TokenClassificationDataset(**dataset_config)

def load_wandb(config):
    """
    初始化wandb（不使用PyTorch Lightning的WandbLogger）
    纯PyTorch版本：直接使用wandb.init
    """
    wandb_config = config.setting.wandb_config
    
    # 直接使用wandb而不是WandbLogger
    wandb.init(
        project=wandb_config.project,
        name=wandb_config.name,
        config=dict(config)
    )
    
    return wandb


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)
    
    if "kwargs" in model_config.keys():
        kwargs = model_config.pop('kwargs')
    else:
        kwargs = {}
        
    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)
    
    if "kwargs" in dataset_config.keys():
        kwargs = dataset_config.pop('kwargs')
    else:
        kwargs = {}
        
    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# def load_plugins():
#     config = get_config()
#     # initialize plugins
#     plugins = []
#
#     if "Trainer_plugin" not in config.keys():
#         return plugins
#
#     if not config.Trainer.logger:
#         if hasattr(config.Trainer_plugin, "LearningRateMonitor"):
#             config.Trainer_plugin.pop("LearningRateMonitor", None)
#
#     if not config.Trainer.enable_checkpointing:
#         if hasattr(config.Trainer_plugin, "ModelCheckpoint"):
#             config.Trainer_plugin.pop("ModelCheckpoint", None)
#
#     for plugin, kwargs in config.Trainer_plugin.items():
#         plugins.append(eval(plugin)(**kwargs))
#
#     return plugins


# load_strategy 函数已移除，不再需要 PyTorch Lightning 的 strategy

# load_trainer 函数已被 PurePyTorchTrainer 替代
# 为了保持向后兼容，创建一个简单的包装函数
def load_trainer(config):
    """
    加载训练器（纯PyTorch版本）
    返回 PurePyTorchTrainer 而不是 pl.Trainer
    """
    from saprot.scripts.training_pure_pytorch import PurePyTorchTrainer
    return PurePyTorchTrainer(config)