# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/6

import logging
import re
import warnings

import torch
from datetime import datetime, timedelta


def calc_recall(sort_lists, batch_size, topk=10):
    recall_result = torch.sum(sort_lists < topk) / batch_size
    return recall_result


def calc_ndcg(sort_lists, batch_size, topk=10):
    hit = sort_lists < topk
    ndcg_score = hit * (1 / torch.log2(sort_lists + 2))
    ndcg_result = torch.sum(ndcg_score) / batch_size
    return ndcg_result

def calc_matrix(predicts, labels):
    predicts_sort = torch.argsort(predicts, dim=-1, descending=True)
    diff = predicts_sort - labels.reshape(-1, 1)
    sort_index = torch.argmax((diff == 0).type_as(diff), dim=-1)
    return sort_index, predicts.shape[0]

def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def get_gpu_usage(device=None):
    r"""Return the reserved memory and total memory of given device in a string.
    Args:
        device: cuda.device. It is the device that the model run on.

    Returns:
        str: it contains the info about reserved memory and total memory of given device.
    """

    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return "{:.2f} G/{:.2f} G".format(reserved, total)

def get_current_time_info():
    # 获取当前 UTC 时间
    now_utc = datetime.utcnow()
    # 将 UTC 时间转换为北京时间
    beijing_time = now_utc + timedelta(hours=8)
    # 格式化时间
    time_info = beijing_time.strftime("%Y-%m-%d_%H-%M-%S")
    return time_info

def get_logger(name, log_file_path):

    # 定义一个移除 ANSI 转义序列的过滤器
    class RemoveColorFilter(logging.Filter):
        def filter(self, record):
            # 使用正则表达式移除 ANSI 转义序列
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            if hasattr(record, 'msg'):
                record.msg = ansi_escape.sub('', record.msg)
            if hasattr(record, 'message'):
                record.message = ansi_escape.sub('', record.message)
            return True

    class CustomLogger:
        def __init__(self, logger_name, log_file):
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.INFO)

            self.file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8', delay=False)
            self.file_handler.setLevel(logging.INFO)

            formatter = logging.Formatter('%(message)s')
            self.file_handler.addFilter(RemoveColorFilter())
            self.file_handler.setFormatter(formatter)
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)

            self.logger.addHandler(stream_handler)
            self.logger.addHandler(self.file_handler)

        def info(self, message):
            self.logger.info(message)

        def close(self):
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()

    return CustomLogger(name, log_file_path)


def get_dataloader(config, dataset_class, num_workers = 4, only_train=False):
    from torch.utils.data import DataLoader
    dataset_path = config['dataset_path']
    max_len = config['max_len']
    batch_size = config['batch_size']
    if 'shuffle' not in config.keys():
        shuffle = False
    else:
        shuffle = config['shuffle']
    if 'large_file_mode' in config.keys():
        large_file_mode = config['large_file_mode']
        if 'is_linecache' in config.keys():
            is_linecache = config['is_linecache']
        else:
            is_linecache = False
    else:
        large_file_mode = False
        is_linecache = False
        if 'is_linecache' in config.keys() and config['is_linecache'] is True:
            warnings.warn(f"When 'large_file_made' is False, 'is_linecache' is not working!")

    train_dataset = dataset_class(root_path=dataset_path, max_len=max_len, split='train', large_file_mode=large_file_mode, is_linecache=is_linecache)
    if only_train is False:
        val_dataset = dataset_class(root_path=dataset_path, max_len=max_len, split='eval', large_file_mode=large_file_mode)
        test_dataset = dataset_class(root_path=dataset_path, max_len=max_len, split='test', large_file_mode=large_file_mode)

    user_num = train_dataset.user_num
    item_num = train_dataset.item_num
    config['user_num'] = user_num
    config['item_num'] = item_num

    if torch.cuda.is_available() and ('pin_memory' in config.keys() and config['pin_memory'] is True):
        pin_memory = True
    else:
        pin_memory = False
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=False, pin_memory=pin_memory)
    if only_train is False:
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=False, pin_memory=pin_memory)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=False, pin_memory=pin_memory)
    if only_train:
        return train_dataloader
    else:
        return train_dataloader, val_dataloader, test_dataloader

def get_config_str(config):
    final_str = 'Config: \n'
    for key in config.keys():
        final_str += f'{key} = {config[key]} \n'
    return final_str