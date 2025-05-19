# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2024/12/15

import torch
import torch.nn as nn

try:
    from .encoder.transformer import TransformerEncoder
    from .abstract import AbstractSRModel
except:
    from encoder.transformer import TransformerEncoder
    from abstract import AbstractSRModel

class SASRecT(AbstractSRModel):
    def __init__(self, config):
        super(SASRecT, self).__init__(config)
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.att_dropout_prob = config["att_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = float(config["layer_norm_eps"])

        # self.item_embedding = None
        feature_dim = self.get_text_embedding(config['text_embed_path'])
        # self.item_embedding_bias = nn.Parameter(torch.zeros(self.item_num + 1, self.hidden_size), requires_grad=True)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )
        self.position_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.att_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def get_text_embedding(self, path):
        text_embed = torch.load(path)
        text_embed = torch.cat([torch.zeros(1, text_embed.shape[-1]), text_embed], dim=0)
        self.register_buffer('item_embedding', text_embed)
        # self.item_embedding = nn.Parameter(text_embed)
        return text_embed.shape[-1]

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # item_emb = self.adapter(self.item_embedding[item_seq]) + self.item_embedding_bias[item_seq]
        item_emb = self.adapter(self.item_embedding[item_seq])
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, item_seq, item_seq_len, labels):
        """
        计算损失
        :param item_seq:
        :param item_seq_len:
        :param labels:
        :return:
        """
        seq_output = self(item_seq, item_seq_len)
        # test_item_emb = self.adapter(self.item_embedding) + self.item_embedding_bias
        test_item_emb = self.adapter(self.item_embedding)
        # test_item_emb = self.bias_linear(test_item_emb)
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
        # test_item_emb = self.adapter(self.item_embedding) + self.item_embedding_bias
        test_item_emb = self.adapter(self.item_embedding)
        # test_item_emb = self.bias_linear(test_item_emb)
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

    def pre_compile(self):
        self.trm_encoder = torch.compile(self.trm_encoder)