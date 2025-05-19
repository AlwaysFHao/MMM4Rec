# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/8

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class AbstractSRModel(ABC, nn.Module):
    def __init__(self, config):
        super(AbstractSRModel, self).__init__()
        self.item_num = config["item_num"]
        self.user_num = config["user_num"]
        self.seq_len = config["max_len"]

    @staticmethod
    def _init_weights(module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @staticmethod
    def get_attention_mask(item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = item_seq != 0
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(
                extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1))
            )
        extended_attention_mask = torch.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    @staticmethod
    def gather_indexes(output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    @staticmethod
    def calc_matrix(predicts, labels):
        predicts_sort = torch.argsort(predicts, dim=-1, descending=True)
        diff = predicts_sort - labels.reshape(-1, 1)
        sort_index = torch.argmax((diff == 0).type_as(diff), dim=-1)
        return sort_index, predicts.shape[0]

    @staticmethod
    def mask_visited_items_efficient(item_seq, scores):
        """
        去除用户已经交互过的项目
        """
        # 创建visited_mask [batch_size, item_num]
        visited_mask = torch.zeros_like(scores, dtype=torch.bool).to(item_seq.device)

        # 生成所有可能的item索引
        batch_size, seq_len = item_seq.shape
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len).to(item_seq.device)

        # 获取所有非padding的(batch_idx, item_idx)对
        non_padding = (item_seq != 0)
        batch_idx = batch_idx[non_padding]
        item_idx = item_seq[non_padding]

        # 标记访问过的item
        visited_mask[batch_idx, item_idx] = True

        # 应用掩码
        scores = scores.masked_fill(visited_mask.to(item_seq.device), float('-inf'))

        return scores

    @abstractmethod
    def training_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    def pre_compile(self):
        """
        Pre compile the model submodule her.
        """
        pass