# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/1/6

import torch
import torch.nn as nn

try:
    from .abstract import AbstractSRModel
except:
    from abstract import AbstractSRModel

class GRU4Rec(AbstractSRModel):

    def __init__(self, config):
        super(GRU4Rec, self).__init__(config)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
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