# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/14
import json
import os

# huggingface镜像，如不需要可注释
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
try:
    from .preprocess_dataset import AmazonPreprocessDataset, collate_fn
except:
    from preprocess_dataset import AmazonPreprocessDataset, collate_fn


def get_clip_dataloader(path, batch_size=64):
    dataset = AmazonPreprocessDataset(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False,
                            collate_fn=collate_fn)
    return dataloader

def get_clip_model(name_or_path):
    clip_model = CLIPModel.from_pretrained(pretrained_model_name_or_path=name_or_path)
    clip_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=name_or_path)
    return clip_model, clip_processor


class ClipExtractFeature(object):
    def __init__(self,dataset_path,
                 clip_model_name_or_path="openai/clip-vit-base-patch32",
                 batch_size=64):
        super(ClipExtractFeature, self).__init__()
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dataloader = get_clip_dataloader(dataset_path, batch_size)
        self.clip_model, self.clip_processor = get_clip_model(clip_model_name_or_path)

    def __call__(self, feature_save_path):
        with torch.no_grad():
            self.clip_model.eval()
            self.clip_model.to(self.device)

            texts_lst = []
            images_lst = []
            print('Start extracting features')
            for (texts, images) in tqdm(self.dataloader):
                inputs = self.clip_processor(text=texts,
                                             images=images,
                                             return_tensors='pt',
                                             padding='max_length',
                                             max_length=77,
                                             truncation=True).to(self.device)
                outputs = self.clip_model(**inputs)
                # [batch, embedding_dim]
                texts_embedding = outputs.text_embeds
                images_embedding = outputs.image_embeds

                texts_lst.append(texts_embedding.cpu())
                images_lst.append(images_embedding.cpu())

            texts_all = torch.cat(texts_lst, dim=0)
            images_all = torch.cat(images_lst, dim=0)

            torch.save(texts_all, os.path.join(feature_save_path, 'text_feat.pth'))
            torch.save(images_all, os.path.join(feature_save_path, 'image_feat.pth'))



if __name__ == '__main__':
    os.chdir('../')
    data_path = '../processed/Beauty/item.jsonl'
    # model_name = "openai/clip-vit-large-patch14"
    model_name = "openai/clip-vit-base-patch32"
    api = ClipExtractFeature(data_path, model_name, batch_size=128)
    api('../processed/Beauty')