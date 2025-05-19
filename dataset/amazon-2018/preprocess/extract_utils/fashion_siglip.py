# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/16
import os

# huggingface镜像，如不需要可注释
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
try:
    from .preprocess_dataset import AmazonPreprocessDataset, collate_fn
except:
    from preprocess_dataset import AmazonPreprocessDataset, collate_fn


def get_siglip_dataloader(path, batch_size=64):
    dataset = AmazonPreprocessDataset(path=path)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4,
                            drop_last=False,
                            collate_fn=collate_fn)
    return dataloader

def get_siglip_model(name_or_path):
    model = AutoModel.from_pretrained(name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
    return model, processor


class FashionSigLipExtractFeature(object):
    def __init__(self,dataset_path,
                 siglip_model_name_or_path="Marqo/marqo-fashionSigLIP",
                 batch_size=64):
        super(FashionSigLipExtractFeature, self).__init__()
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

                processed = self.siglip_processor(text=texts, images=images, padding='max_length', return_tensors="pt", truncation=True).to(self.device)

                images_embedding = self.siglip_model.get_image_features(processed['pixel_values'], normalize=True)
                texts_embedding = self.siglip_model.get_text_features(processed['input_ids'], normalize=True)

                texts_lst.append(texts_embedding.cpu())
                images_lst.append(images_embedding.cpu())

            texts_all = torch.cat(texts_lst, dim=0)
            images_all = torch.cat(images_lst, dim=0)

            torch.save(texts_all, os.path.join(feature_save_path, 'text_feat.pth'))
            torch.save(images_all, os.path.join(feature_save_path, 'image_feat.pth'))



if __name__ == '__main__':
    os.chdir('../')
    data_path = '../processed/Beauty/item.jsonl'
    model_name = "Marqo/marqo-fashionSigLIP"
    api = FashionSigLipExtractFeature(data_path, model_name, batch_size=128)
    api('../processed/Beauty')