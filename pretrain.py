# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/1/17

import os

import torch

from data.amazon_dataset import AmazonSequentialDataset
from model.mmm4rec import MMM4Rec

import yaml
from lightning.pytorch import seed_everything
from trainer.pretrain_trainer import PreTrainer
from trainer.utils import get_dataloader

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    config_yaml_file_path = os.path.join('configs', 'config_mmm4rec_FHCKM.yaml')
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # seed_everything(config['seed'], workers=True)

    train_dataloader = get_dataloader(config=config, dataset_class=AmazonSequentialDataset, num_workers=0, only_train=True)

    model = MMM4Rec(config)
    from model.encoder.ssdv2 import TiSSD, TiSSDLayer, CoTiSSD, CoTiSSDLayer, FeedForward
    save_model_classes = [TiSSD, TiSSDLayer, CoTiSSD, CoTiSSDLayer, FeedForward]

    optimizer_lr = config['optimizer_lr']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.NAdam(model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)

    if 'checkpoint' in config.keys():
        checkpoint = config['checkpoint']
    else:
        checkpoint = None

    trainer = PreTrainer(model=model,
                      optimizer=optimizer,
                      config=config,
                      checkpoint=checkpoint,
                      save_model_classes=save_model_classes)

    trainer.fit(train_dataloader=train_dataloader)
