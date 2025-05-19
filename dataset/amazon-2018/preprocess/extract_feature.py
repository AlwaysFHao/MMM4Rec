# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/14
import argparse
import os

import torch

from extract_utils.clip_and_bert import ClipBertExtractFeature
from extract_utils.clip import ClipExtractFeature
from extract_utils.siglip import SigLipExtractFeature
from extract_utils.blipv2 import Blip2ExtractFeature
from extract_utils.blip import BlipExtractFeature
from extract_utils.siglip_and_roberta import SiglipRoBertaExtractFeature

if __name__ == '__main__':
    model = 'SigLIP'  # [SigLIP, CLIP, BLIP, BLIP2, SigLIP_Roberta]
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub_dataset', default=[], nargs='*', help='datasets to be processed')
    args = parser.parse_args()
    preprocess_dataset = args.sub_dataset
    if len(preprocess_dataset) == 0:
        preprocess_dataset = None

    print(f"The sub dataset is as follows:{os.listdir('../processed/')}")
    for sub_dataset in os.listdir('../processed/'):
        if sub_dataset in preprocess_dataset if preprocess_dataset is not None else True:
            print(f'\n-----------------Processing {sub_dataset}-----------------\n')
            data_path = os.path.join('../processed', sub_dataset, 'item.jsonl')
            if model == 'CLIP':
                api = ClipExtractFeature(data_path, "openai/clip-vit-base-patch32", batch_size=512)
            elif model == 'SigLIP':
                api = SigLipExtractFeature(data_path, "google/siglip-base-patch16-224", batch_size=512)
            elif model == 'BLIP':
                api = BlipExtractFeature(data_path, "Salesforce/blip-itm-base-coco", batch_size=128)
            elif model == 'BLIP2':
                api = Blip2ExtractFeature(data_path, "Salesforce/blip2-itm-vit-g", batch_size=64)
            elif model == 'SigLIP_Roberta':
                api = SiglipRoBertaExtractFeature(data_path, batch_size=512)
            elif model == 'CLIP_Bert':
                api = ClipBertExtractFeature(data_path, batch_size=64)
            else:
                raise ValueError(f'{model} is not available!')
            api(os.path.join('../processed', sub_dataset))
            torch.cuda.empty_cache()
    print('Finish')
