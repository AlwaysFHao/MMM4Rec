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
from transformers import CLIPModel, CLIPProcessor, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer

try:
    from .preprocess_dataset import AmazonPreprocessDataset, collate_fn
except:
    from preprocess_dataset import AmazonPreprocessDataset, collate_fn

def get_dataloader(path, batch_size=64):
    dataset = AmazonPreprocessDataset(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False,
                            collate_fn=collate_fn)
    return dataloader

def get_model(vision_name_or_path, text_name_or_path):
    vision_model = CLIPModel.from_pretrained(pretrained_model_name_or_path=vision_name_or_path)
    vision_processor = CLIPProcessor.from_pretrained(pretrained_model_name_or_path=vision_name_or_path)
    # text_model = RobertaModel.from_pretrained(pretrained_model_name_or_path=text_name_or_path)
    # text_tokenizer = RobertaTokenizer.from_pretrained(pretrained_model_name_or_path=text_name_or_path)

    text_model = AutoModel.from_pretrained(pretrained_model_name_or_path=text_name_or_path)
    text_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=text_name_or_path)
    return vision_model, text_model, vision_processor, text_tokenizer


class ClipRoBertaExtractFeature(object):
    def __init__(self, dataset_path,
                 vision_model_name_or_path="openai/clip-vit-base-patch32",
                 text_model_name_or_path="FacebookAI/roberta-base",
                 batch_size=64):
        super(ClipRoBertaExtractFeature, self).__init__()
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dataloader = get_dataloader(dataset_path, batch_size)
        self.vision_model, self.text_model, self.vision_processor, self.text_tokenizer = get_model(vision_model_name_or_path, text_model_name_or_path)

    def __call__(self, feature_save_path):
        with torch.no_grad():
            self.vision_model.to(self.device)
            self.text_model.to(self.device)
            self.vision_model.eval()
            self.text_model.eval()

            texts_lst = []
            images_lst = []
            print('Start extracting features')
            for (texts, images) in tqdm(self.dataloader):
                vision_inputs = self.vision_processor(images=images, return_tensors='pt').to(self.device)
                texts_inputs = self.text_tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts,
                                                                truncation=True,
                                                                padding=True,
                                                                max_length=512,
                                                                return_tensors='pt').to(self.device)
                texts_embedding = self.text_model(**texts_inputs)[1]
                images_embedding = self.vision_model.get_image_features(**vision_inputs)

                texts_lst.append(texts_embedding.cpu())
                images_lst.append(images_embedding.cpu())

            texts_all = torch.cat(texts_lst, dim=0)
            images_all = torch.cat(images_lst, dim=0)

            torch.save(texts_all, os.path.join(feature_save_path, 'tmp_text_feat.pth'))
            torch.save(images_all, os.path.join(feature_save_path, 'tmp_image_feat.pth'))



if __name__ == '__main__':
    os.chdir('../')
    data_path = '../processed/Scientific/item.jsonl'
    vision_model_name = "openai/clip-vit-base-patch32"
    # text_model_name = "FacebookAI/roberta-base"
    text_model_name = "bert-base-uncased"
    api = ClipRoBertaExtractFeature(data_path, vision_model_name, text_model_name, batch_size=128)
    api('../processed/Scientific')