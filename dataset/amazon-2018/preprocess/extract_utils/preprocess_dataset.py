# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/14
import json
import os.path

from torch.utils.data import Dataset
from PIL import Image

class AmazonPreprocessDataset(Dataset):
    def __init__(self, path):
        super(AmazonPreprocessDataset, self).__init__()
        with open(path, 'r') as f:
            self.info_file = json.load(f)

    def __len__(self):
        return len(self.info_file.keys())

    def __getitem__(self, item):
        item += 1
        if self.info_file[str(item)]['vision'] is None:
            image = Image.new('RGB', (200, 200), color='white')
        else:
            pic_path = self.info_file[str(item)]['vision'].replace(f'./dataset/amazon-2018/preprocess{os.path.sep}', '')
            image = Image.open(pic_path).convert('RGB')
        text_path = self.info_file[str(item)]['text'].replace(f'./dataset/amazon-2018/preprocess{os.path.sep}', '')
        with open(text_path, 'r') as f:
            text = f.read()
        return text, image


def collate_fn(data):
    texts = []
    pics = []
    for temp_data in data:
        texts.append(temp_data[0])
        pics.append(temp_data[1])

    return texts, pics