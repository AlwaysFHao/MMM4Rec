# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/22

import argparse
import torch

from data.amazon_dataset import AmazonSequentialDataset
from model.mmm4rec import MMM4Rec

import yaml
from lightning.pytorch import seed_everything
from trainer.trainer import Trainer
from trainer.utils import get_dataloader

if torch.cuda.is_available():
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MMM4Rec Test Script')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration YAML file')
    args = parser.parse_args()

    config_yaml_file_path = args.config
    with open(config_yaml_file_path, 'r') as stream:
        config = yaml.safe_load(stream)

    if 'float32_precise' in config.keys():
        if config['float32_precise'] == 'medium':
            torch.set_float32_matmul_precision('medium')
        elif config['float32_precise'] == 'high':
            torch.set_float32_matmul_precision('high')
    seed_everything(config['seed'], workers=True)

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(config=config,
                                                                       dataset_class=AmazonSequentialDataset,
                                                                       num_workers=0)

    model = MMM4Rec(config)
    from model.encoder.ssd import TiSSD, TiSSDLayer, CoTiSSD, CoTiSSDLayer, FeedForward

    save_model_classes = [TiSSD, TiSSDLayer, CoTiSSD, CoTiSSDLayer, FeedForward]

    optimizer_lr = config['optimizer_lr']
    weight_decay = config['weight_decay']

    # for _ in model.encoder_ssd_layers.parameters():
    #     _.requires_grad = False
    #
    # for _ in model.decoder_ssd_layers.parameters():
    #     _.requires_grad = False

    optimizer = torch.optim.NAdam(model.parameters(), lr=optimizer_lr, weight_decay=weight_decay)

    if 'checkpoint' in config.keys():
        checkpoint = config['checkpoint']
    else:
        checkpoint = None

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      config=config,
                      checkpoint=checkpoint,
                      save_model_classes=save_model_classes)

    trainer.test(test_dataloader=test_dataloader)