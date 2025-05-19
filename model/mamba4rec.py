# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/2/17


import torch
import torch.nn as nn
from mamba_ssm import Mamba

try:
    from .abstract import AbstractSRModel
except:
    from abstract import AbstractSRModel

class Mamba4Rec(AbstractSRModel):
    def __init__(self, config):
        super(Mamba4Rec, self).__init__(config)
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)


    def forward(self, item_seq, item_seq_len):

        item_emb = self.item_embedding(item_seq)
        item_emb = self.dropout(item_emb)
        item_emb = self.LayerNorm(item_emb)

        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](item_emb)

        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, item_seq, item_seq_len, labels):
        """
        计算损失
        :param item_seq:
        :param item_seq_len:
        :param labels:
        :return:
        """
        seq_output = self(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len):
        """
        全排序预测
        :param item_seq:
        :param item_seq_len:
        :return:
        """
        seq_output = self(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return scores

    def training_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        loss = self.calculate_loss(item_seq=user_item_seq, item_seq_len=length, labels=labels)
        return loss

    def validation_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        scores = self.full_sort_predict(item_seq=user_item_seq, item_seq_len=length)
        sort_index, batch = self.calc_matrix(scores, labels)
        return sort_index, batch


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        hidden_states = self.mamba(input_tensor)
        if self.num_layers == 1:  # one Mamba layer without residual connection
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:  # stacked Mamba layers with residual connections
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states