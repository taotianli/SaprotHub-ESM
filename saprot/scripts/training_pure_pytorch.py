"""
纯PyTorch训练脚本，不使用PyTorch Lightning
用于替换原有的training.py中的PyTorch Lightning实现
"""

import sys
import os

# 在导入其他库之前设置环境变量，避免accelerate相关的numpy兼容性问题
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'
# 禁用HuggingFace的离线模式检查，避免不必要的网络请求
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from pathlib import Path
import json
from tqdm import tqdm
import copy

current_file = os.path.abspath(__file__)
saprot_dir = os.path.dirname(current_file)
colabsaprot_dir = os.path.dirname(saprot_dir)
sys.path.append(colabsaprot_dir)

import yaml
import argparse
from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *


class PurePyTorchTrainer:
    """纯PyTorch训练器，替代PyTorch Lightning的Trainer"""
    
    def __init__(self, config):
        """
        初始化训练器
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_epochs = config.Trainer.get('max_epochs', 10)
        self.accelerator = config.Trainer.get('accelerator', 'auto')
        self.accumulate_grad_batches = config.Trainer.get('accumulate_grad_batches', 1)
        self.gradient_clip_val = config.Trainer.get('gradient_clip_val', None)
        self.log_every_n_steps = config.Trainer.get('log_every_n_steps', 50)
        self.enable_checkpointing = config.Trainer.get('enable_checkpointing', True)
        
        # 确定设备
        if self.accelerator == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif self.accelerator == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # print(f"训练设备: {self.device}")
        
        # 初始化日志记录
        self.logger = config.Trainer.get('logger', None)
        self.global_step = 0
        self.current_epoch = 0
        
    def fit(self, model, datamodule):
        """
        训练模型
        Args:
            model: 模型实例
            datamodule: 数据模块实例
        """
        # print("开始训练...")
        
        # 将模型移到设备
        model = model.to(self.device)
        
        # 设置数据加载器
        datamodule.setup('fit')
        train_dataloader = datamodule.train_dataloader()
        val_dataloader = datamodule.val_dataloader()
        
        # 设置ESM3模型到数据集（如果需要）
        if hasattr(model, '_set_esm_model_to_datasets'):
            self._inject_trainer_to_model(model, datamodule)
            model._set_esm_model_to_datasets()
        
        # 获取优化器和学习率调度器
        optimizer_config = model.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        lr_scheduler = optimizer_config['lr_scheduler']['scheduler']
        
        # 训练循环
        best_metric = None
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            
            # 训练阶段
            model.train()
            train_loss = self._train_epoch(model, train_dataloader, optimizer, lr_scheduler)
            # print(f"训练损失: {train_loss:.4f}")
            
            # 验证阶段
            if val_dataloader is not None:
                model.eval()
                # 同步 model.step 与 trainer.global_step
                if hasattr(model, 'step'):
                    model.step = self.global_step
                val_metrics = self._validate_epoch(model, val_dataloader)
                # print(f"验证指标: {val_metrics}")
                
                # 检查是否保存模型
                if self.enable_checkpointing and hasattr(model, 'check_save_condition'):
                    # 假设我们使用验证准确率作为保存指标
                    if 'valid_acc' in val_metrics:
                        model.check_save_condition(val_metrics['valid_acc'], mode='max')
            
            # 更新epoch计数
            if hasattr(model, 'on_train_epoch_end'):
                model.on_train_epoch_end()
        
        # print("\n训练完成!")
        
        # 确保至少保存最后一个epoch的模型（即使验证指标不是最佳的）
        if self.enable_checkpointing and model.save_path is not None:
            save_path = model.save_path
            if not save_path.endswith('.pt'):
                save_path = save_path + '.pt'
            
            # 检查是否已经保存过模型
            if not os.path.exists(save_path):
                print(f"\n⚠️ 没有保存过最佳模型（可能验证指标一直没有改进），强制保存最后的模型...")
                model.save_checkpoint(save_path, save_info=None, save_weights_only=model.save_weights_only)
                
                # 验证文件大小
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path) / (1024 * 1024)
                    print(f"✅ 模型已保存到: {save_path} ({file_size:.2f} MB)")
                else:
                    print(f"❌ 模型保存失败")
            else:
                file_size = os.path.getsize(save_path) / (1024 * 1024)
                print(f"\n✅ 最佳模型已保存到: {save_path} ({file_size:.2f} MB)")
        
    def test(self, model, datamodule):
        """
        测试模型
        Args:
            model: 模型实例
            datamodule: 数据模块实例
        """
        # print("开始测试...")
        
        # 将模型移到设备
        model = model.to(self.device)
        model.eval()
        
        # 设置数据加载器
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        
        # 设置ESM3模型到数据集（如果需要）
        if hasattr(model, '_set_esm_model_to_datasets'):
            self._inject_trainer_to_model(model, datamodule)
            model._set_esm_model_to_datasets()
        
        # 测试循环
        test_metrics = self._test_epoch(model, test_dataloader)
        # print(f"测试指标: {test_metrics}")
        
        return test_metrics
    
    def _inject_trainer_to_model(self, model, datamodule):
        """注入trainer引用到模型，用于兼容原有的代码"""
        # 创建一个简单的trainer对象，包含datamodule引用
        class SimpleTrainer:
            def __init__(self, datamodule):
                self.datamodule = datamodule
        
        model.trainer = SimpleTrainer(datamodule)
    
    def _train_epoch(self, model, dataloader, optimizer, lr_scheduler):
        """训练一个epoch"""
        total_loss = 0.0
        num_batches = 0
        accumulation_counter = 0
        
        # 创建进度条
        pbar = tqdm(dataloader, desc="训练中")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            inputs, labels = batch
            outputs = model(**inputs)
            loss = model.loss_func('train', outputs, labels)
            
            # 缩放损失（用于梯度累积）
            loss = loss / self.accumulate_grad_batches
            
            # 反向传播
            loss.backward()
            
            accumulation_counter += 1
            
            # 梯度累积
            if accumulation_counter % self.accumulate_grad_batches == 0:
                # 梯度裁剪
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_val)
                
                # 优化器步骤
                optimizer.step()
                optimizer.zero_grad()
                
                # 学习率调度器步骤
                if lr_scheduler is not None:
                    lr_scheduler.step()
                
                self.global_step += 1
                
                # 记录日志
                if self.global_step % self.log_every_n_steps == 0 and self.logger:
                    log_dict = {
                        'train_loss': loss.item() * self.accumulate_grad_batches,
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'epoch': self.current_epoch
                    }
                    if hasattr(self.logger, 'log_metrics'):
                        self.logger.log_metrics(log_dict, step=self.global_step)
            
            total_loss += loss.item() * self.accumulate_grad_batches
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item() * self.accumulate_grad_batches:.4f}'})
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate_epoch(self, model, dataloader):
        """验证一个epoch"""
        # 调用模型的验证开始钩子
        if hasattr(model, 'on_validation_epoch_start'):
            model.on_validation_epoch_start()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="验证中")
            for batch in pbar:
                # 将数据移到设备
                batch = self._move_batch_to_device(batch)
                
                # 前向传播
                inputs, labels = batch
                outputs = model(**inputs)
                loss = model.loss_func('valid', outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        # 调用模型的验证结束钩子
        metrics = {}
        if hasattr(model, 'on_validation_epoch_end'):
            model.on_validation_epoch_end()
            # 获取验证指标
            if hasattr(model, 'get_log_dict'):
                metrics = model.get_log_dict('valid')
                if 'valid_loss' not in metrics:
                    metrics['valid_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        else:
            metrics['valid_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def _test_epoch(self, model, dataloader):
        """测试一个epoch"""
        # 调用模型的测试开始钩子
        if hasattr(model, 'on_test_epoch_start'):
            model.on_test_epoch_start()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="测试中")
            for batch in pbar:
                # 将数据移到设备
                batch = self._move_batch_to_device(batch)
                
                # 前向传播
                inputs, labels = batch
                outputs = model(**inputs)
                loss = model.loss_func('test', outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        # 调用模型的测试结束钩子
        metrics = {}
        if hasattr(model, 'on_test_epoch_end'):
            model.on_test_epoch_end()
            # 获取测试指标
            if hasattr(model, 'get_log_dict'):
                metrics = model.get_log_dict('test')
                if 'test_loss' not in metrics:
                    metrics['test_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        else:
            metrics['test_loss'] = total_loss / num_batches if num_batches > 0 else 0.0
        
        return metrics
    
    def _move_batch_to_device(self, batch):
        """将批次数据移到设备"""
        inputs, labels = batch
        
        # 移动inputs
        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in inputs.items()}
        elif torch.is_tensor(inputs):
            inputs = inputs.to(self.device)
        
        # 移动labels
        if isinstance(labels, dict):
            labels = {k: v.to(self.device) if torch.is_tensor(v) else v 
                     for k, v in labels.items()}
        elif torch.is_tensor(labels):
            labels = labels.to(self.device)
        
        return inputs, labels


def finetune_pure_pytorch(config):
    """
    使用纯PyTorch进行微调，不使用PyTorch Lightning
    Args:
        config: 配置字典
    """
    if config.setting.seed:
        setup_seed(config.setting.seed)

    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)
        elif k in os.environ:
            config.setting.os_environ[k] = os.environ[k]

    # 加载模型
    model = my_load_model(config.model)
    
    # 加载数据集
    if str(config.setting.seed):
        config.dataset.seed = config.setting.seed
    data_module = my_load_dataset(config.dataset)
    
    # 创建纯PyTorch训练器
    trainer = PurePyTorchTrainer(config)
    
    # 训练
    trainer.fit(model=model, datamodule=data_module)
    
    # 加载最佳模型并测试
    if model.save_path is not None:
        # print(f"\n从 {model.save_path} 加载最佳模型进行测试...")
        model.load_checkpoint(model.save_path)
        trainer.test(model=model, datamodule=data_module)


def main(args):
    """主函数"""
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    if config.setting.seed:
        setup_seed(config.setting.seed)

    # 设置环境变量
    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)
        elif k in os.environ:
            config.setting.os_environ[k] = os.environ[k]

    finetune_pure_pytorch(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="运行配置文件", type=str, required=True)
    args = parser.parse_args()
    main(args)

