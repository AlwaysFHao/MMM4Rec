# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/6

import json
import linecache
import os

import torch
from torch.utils.data import Dataset, DataLoader

import jsonlines


class AmazonSequentialDataset(Dataset):
    def __init__(self, root_path, max_len, split='train', large_file_mode: bool = True, is_linecache: bool = True):
        super(AmazonSequentialDataset, self).__init__()
        assert split in ['train', 'eval', 'test'], f"请输入正确的数据集类别，{split}不在['train', 'eval', 'test']中！"
        self.seq_dir = os.path.join(root_path, f'{split}_seq.jsonl')
        self.large_file_mode = large_file_mode
        self.is_linecache = is_linecache
        if self.large_file_mode:
            if self.is_linecache:
                with open(self.seq_dir, 'r', encoding='utf-8') as file:
                    self.dataset_count = sum(1 for _ in file)
            else:
                self.offsets = []
                self.dataset_count = 0
                # 预计算每行的偏移量
                with open(self.seq_dir, 'rb') as f:
                    while True:
                        offset = f.tell()
                        line = f.readline()
                        if not line:
                            break
                        self.dataset_count += 1
                        self.offsets.append(offset)

                self.length = len(self.offsets)
        else:
            self.seq_list = []
            with open(self.seq_dir, 'r', encoding="utf8") as f:
                for item in jsonlines.Reader(f):
                    self.seq_list.append(item)


        self.max_len = max_len
        self.user_num = 0
        self.item_num = 0
        if split == 'train':
            with open(os.path.join(root_path, 'user2id.jsonl'), 'r', encoding="utf8") as f:
                for _ in jsonlines.Reader(f):
                    self.user_num += 1
            with open(os.path.join(root_path, 'item2id.jsonl'), 'r', encoding="utf8") as f:
                for _ in jsonlines.Reader(f):
                    self.item_num += 1


    def __len__(self):
        return self.dataset_count if self.large_file_mode else len(self.seq_list)

    def __getitem__(self, idx):
        if self.large_file_mode:
            if self.is_linecache:
                # linecache.clearcache()
                # 使用 linecache 读取特定行
                line = linecache.getline(self.seq_dir, idx + 1)
                # 使用 jsonlines 解析该行
                seq = json.loads(line)
            else:
                # 每次读取时打开文件，确保多进程安全
                with open(self.seq_dir, 'rb') as f:
                    f.seek(self.offsets[idx])
                    line = f.readline().decode('utf-8').strip()

                # 使用 jsonlines 解析该行
                seq = json.loads(line)
                del line
        else:
            seq = self.seq_list[idx]

        user = seq[0]
        item_seq = [item[0] for item in seq[1]]
        time_stamps = [item[1] for item in seq[1]]

        if len(item_seq) > self.max_len:
            padding_mask = torch.ones(self.max_len) == 0
            label = item_seq[-1]
            time_stamps = time_stamps[-(self.max_len + 1): -1]
            item_seq = item_seq[-(self.max_len + 1): -1]
            length = self.max_len
        else:
            label = item_seq.pop()
            padding_mask = torch.cat((torch.ones(len(item_seq)), torch.zeros(self.max_len - len(item_seq)))) == 0
            length = len(item_seq)
            item_seq.extend([0 for _ in range(self.max_len - len(item_seq))])
            time_stamps.extend([0 for _ in range(self.max_len - len(time_stamps))])
        del seq
        user = torch.tensor(user)
        item_seq = torch.tensor(item_seq)
        label = torch.tensor(label)
        length = torch.tensor(length)
        time_stamps = torch.tensor(time_stamps)

        return user, item_seq, label, length, padding_mask, time_stamps


if __name__ == '__main__':
    dataset = AmazonSequentialDataset(root_path='./dataset/amazon/processed/Beauty', max_len=50, split='eval')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    for i, data in enumerate(dataloader):
        print(data)
