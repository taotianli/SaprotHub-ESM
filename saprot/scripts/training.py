import sys
import os
current_file = os.path.abspath(__file__)
saprot_dir = os.path.dirname(current_file)
colabsaprot_dir = os.path.dirname(saprot_dir)
sys.path.append(colabsaprot_dir)

import yaml
import argparse

from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *

# 导入纯PyTorch训练器
from saprot.scripts.training_pure_pytorch import finetune_pure_pytorch, PurePyTorchTrainer



################################################################################
################################## finetune ####################################
################################################################################
def finetune(config):
    """
    使用纯PyTorch训练替代PyTorch Lightning
    """
    # 调用纯PyTorch训练函数
    finetune_pure_pytorch(config)


    ############################################################################
    # if config.setting.run_mode == 'test':
    #     if config.model.kwargs.get("use_lora", False):
    #         if config.model.kwargs.lora_config_path is not None:
    #             config.model.kwargs.lora_inference = True
    #     trainer.test(model=model, datamodule=data_module)
    

def run(config):
    # Initialize a model
    model = load_model(config.model)

    # for i, (name, param) in enumerate(model.named_parameters()):
    #     print(f"{i}: {name}", param.requires_grad, id(param))
    # return

    # Initialize a dataset
    if str(config.setting.seed):
        config.dataset.seed = config.setting.seed
    data_module = load_dataset(config.dataset)

    # Initialize a trainer
    trainer = load_trainer(config)

    # Train and validate
    trainer.fit(model=model, datamodule=data_module)

    # Load best model and test performance
    if model.save_path is not None:
        # adapter和LoRA加载逻辑已注释，因为已禁用LoRA
        # if config.model.kwargs.get("lora_kwargs", None):
        #     # Load LoRA model
        #     if len(getattr(config.model.kwargs.lora_kwargs, "config_list", [])) <= 1:
        #         config.model.kwargs.lora_kwargs.num_lora = 1
        #         config.model.kwargs.lora_kwargs.config_list = [{"lora_config_path": model.save_path}]
        #         
        #     model = load_model(config.model)
        # else:
        #     model.load_checkpoint(model.save_path)
        
        model.load_checkpoint(model.save_path)
        trainer.test(model=model, datamodule=data_module)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    return parser.parse_args()


def main(args):
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    if config.setting.seed:
        setup_seed(config.setting.seed)

    # set os environment variables
    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

        elif k in os.environ:
            # override the os environment variables
            config.setting.os_environ[k] = os.environ[k]

    # Only the root node will print the log
    if config.setting.os_environ.NODE_RANK != 0:
        config.Trainer.logger = False

    run(config)


if __name__ == '__main__':
    main(get_args())
    # with open(args.config, 'r', encoding='utf-8') as r:
    #     config = EasyDict(yaml.safe_load(r))
    
    # finetune(config)