# -*- coding: utf-8 -*-
# @Author : Hao Fan
# @Time : 2024/12/23

import sys
import os
sys.path.append('../../')

import torch

from data.amazon_dataset import AmazonSequentialDataset
from model.mamba4rec_t import Mamba4Rec_T


import yaml
from lightning.pytorch import seed_everything
from trainer.trainer import Trainer
from trainer.utils import get_dataloader

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    config_yaml_file_path = 'config_mamba4rec.yaml'

    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    os.chdir('../../')
    seed_everything(config['seed'], workers=True)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config, dataset_class=AmazonSequentialDataset, num_workers=0)

    model = Mamba4Rec_T(config)
    save_model_classes = []

    optimizer_lr = config['optimizer_lr']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)

    if 'checkpoint' in config.keys():
        checkpoint = config['checkpoint']
    else:
        checkpoint = None

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      config=config,
                      checkpoint=checkpoint,
                      save_model_classes=save_model_classes)

    trainer.fit(train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader)
