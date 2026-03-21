# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/16
import os

# huggingface镜像，如不需要可注释
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor
try:
    from .preprocess_dataset import AmazonPreprocessDataset, collate_fn
except:
    from preprocess_dataset import AmazonPreprocessDataset, collate_fn


def get_siglip_dataloader(path, batch_size=64):
    dataset = AmazonPreprocessDataset(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=6,
                            drop_last=False,
                            collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
    return dataloader

def get_siglip_model(name_or_path):
    clip_model = SiglipModel.from_pretrained(pretrained_model_name_or_path=name_or_path)
    clip_processor = SiglipProcessor.from_pretrained(pretrained_model_name_or_path=name_or_path)

    return clip_model, clip_processor


class SigLipExtractFeature(object):
    def __init__(self,dataset_path,
                 siglip_model_name_or_path="google/siglip-base-patch16-224", # google/siglip-so400m-patch14-384, google/siglip-base-patch16-224
                 batch_size=64):
        super(SigLipExtractFeature, self).__init__()
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.dataloader = get_siglip_dataloader(dataset_path, batch_size)
        self.siglip_model, self.siglip_processor = get_siglip_model(siglip_model_name_or_path)

    def __call__(self, feature_save_path):
        with torch.no_grad():
            self.siglip_model.eval()
            self.siglip_model.to(self.device)

            texts_lst = []
            images_lst = []
            print('Start extracting features')
            for (texts, images) in tqdm(self.dataloader):
                inputs = self.siglip_processor(text=texts,
                                             images=images,
                                             return_tensors='pt',
                                             padding='max_length',
                                             max_length=64,
                                             truncation=True).to(self.device)

                outputs = self.siglip_model(**inputs)
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
    model_name = "google/siglip-base-patch16-224"
    api = SigLipExtractFeature(data_path, model_name, batch_size=128)
    api('../processed/Beauty')