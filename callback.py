# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/6

import inspect
import os


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False

def save_class_code(cls, save_path, logger=None):
    # 获取类的源代码
    source = inspect.getsource(cls)

    # 获取类名
    class_name = cls.__name__

    # 创建一个文件名
    filename = os.path.join(save_path, f"{class_name}.py")

    # 保存到文件
    with open(filename, 'w') as f:
        f.write(source)

    if logger is not None:
        logger.info(f"Class code for '{class_name}' saved to '{filename}'.")



