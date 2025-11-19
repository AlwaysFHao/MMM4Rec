# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/16

import os

# huggingface镜像，如不需要可注释
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import AutoProcessor, BlipForImageTextRetrieval
try:
    from .preprocess_dataset import AmazonPreprocessDataset, collate_fn
except:
    from preprocess_dataset import AmazonPreprocessDataset, collate_fn


def get_blip_dataloader(path, batch_size=64):
    dataset = AmazonPreprocessDataset(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False,
                            collate_fn=collate_fn)
    return dataloader

def get_blip_model(name_or_path):
    clip_model = BlipForImageTextRetrieval.from_pretrained(pretrained_model_name_or_path=name_or_path)
    clip_processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path=name_or_path)
    return clip_model, clip_processor


class BlipExtractFeature(object):
    def __init__(self,dataset_path,
                 blip2_model_name_or_path="Salesforce/blip-itm-base-coco",
                 batch_size=64):
        super(BlipExtractFeature, self).__init__()
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dataloader = get_blip_dataloader(dataset_path, batch_size)
        self.blip_model, self.blip_processor = get_blip_model(blip2_model_name_or_path)

    def __call__(self, feature_save_path):
        with torch.no_grad():
            self.blip_model.eval()
            self.blip_model.to(self.device)

            texts_lst = []
            images_lst = []
            print('Start extracting features')
            for (texts, images) in tqdm(self.dataloader):
                inputs = self.blip_processor(text=texts,
                                             images=images,
                                             return_tensors='pt',
                                             padding='max_length',
                                             max_length=512,
                                             truncation=True).to(self.device)
                outputs = self.blip_model(**inputs)
                # [batch, embedding_dim]
                texts_embedding = outputs.question_embeds[:, 0, :]
                images_embedding = outputs.last_hidden_state[:, 0, :]

                texts_lst.append(texts_embedding.cpu())
                images_lst.append(images_embedding.cpu())

            texts_all = torch.cat(texts_lst, dim=0)
            images_all = torch.cat(images_lst, dim=0)

            torch.save(texts_all, os.path.join(feature_save_path, 'text_feat.pth'))
            torch.save(images_all, os.path.join(feature_save_path, 'image_feat.pth'))



if __name__ == '__main__':
    os.chdir('../')
    data_path = '../processed/Beauty/item.jsonl'
    model_name = "Salesforce/blip-itm-base-coco"
    api = BlipExtractFeature(data_path, model_name, batch_size=128)
    api('../processed/Beauty')