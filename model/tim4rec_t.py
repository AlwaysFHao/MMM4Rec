# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2025/4/11


import torch
import torch.nn as nn
from einops import repeat

try:
    from .encoder.ssdv2 import TiSSDLayer
    from .abstract import AbstractSRModel
except:
    from encoder.ssdv2 import TiSSDLayer
    from abstract import AbstractSRModel

class TiM4Rec_T(AbstractSRModel):
    def __init__(self, config):
        super(TiM4Rec_T, self).__init__(config)
        # Hyperparameters for TiM4Rec
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.time_drop_out = config['time_drop_out']

        # Hyperparameters for SSDLayer
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']
        self.head_dim = config['head_dim']
        self.chunk_size = config['chunk_size']
        self.is_ffn = config['is_ffn']
        self.is_time = config['is_time']
        self.p2p_residual = config['p2p_residual']
        self.norm_eps = float(config['norm_eps'])
        self.is_kai_ming_init = config['is_kai_ming_init']
        self.max_seq_length = config['max_len']
        self.ngroups = config['n_groups']
        assert (self.hidden_size * self.expand) % self.head_dim == 0, \
            f'hidden_size * expand {self.hidden_size * self.expand} can\'t divisible by head_dim {self.head_dim} !'
        self.n_heads = (self.hidden_size * self.expand) // self.head_dim
        # [PAD] has been added to the actual number of items in Recbole
        feature_dim = self.get_text_embedding(config['text_embed_path'])
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )

        self.in_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        if self.is_time:
            # self.time_start_token = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.layer_norm_time = nn.LayerNorm(self.max_seq_length, eps=self.norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.ssd_layers = nn.ModuleList([
            TiSSDLayer(
                d_model=self.hidden_size,
                seq_len=self.max_seq_length,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                time_drop_out=self.time_drop_out,
                is_ffn=self.is_ffn,
                is_time=self.is_time,
                p2p_residual=self.p2p_residual,
                norm_eps=self.norm_eps,
                inner_size=self.hidden_size * 4,
                ngroups=self.ngroups
            ) for _ in range(self.num_layers)
        ])

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def get_text_embedding(self, path):
        text_embed = torch.load(path)
        text_embed = torch.cat([torch.zeros(1, text_embed.shape[-1]), text_embed], dim=0)
        self.register_buffer('item_embedding', text_embed)
        # self.item_embedding = nn.Parameter(text_embed)
        return text_embed.shape[-1]

    def calculate_time_diff(self, time_stamp):
        """
        Calculate the interaction time difference
        :param time_stamp: [batch_size, seq_len]
        :return: [batch_size, seq_len]
        """
        batch_size = time_stamp.shape[0]
        # [batch_size, seq_len - 1]
        time_diff = time_stamp[:, 1:] - time_stamp[:, :-1]
        # add first time diff
        # time_diff = torch.concat([repeat(self.time_start_token, '1 -> b 1', b=batch_size), time_diff], dim=1)
        time_diff = torch.concat([torch.zeros(batch_size, 1).to(time_diff.device), time_diff], dim=1)
        # time_diff = nn.functional.normalize(time_diff, p=2, dim=-1)
        time_diff = self.layer_norm_time(self.dropout(time_diff))
        # [batch_size, seq_len] -> [batch_size, n_heads, seq_len]
        time_diff = repeat(time_diff, 'b l -> b h l', h=self.n_heads)
        return time_diff

    def forward(self, item_seq, item_seq_len, time_stamps):
        item_emb = self.adapter(self.item_embedding[item_seq])
        item_emb = self.dropout(item_emb)
        item_emb = self.in_layer_norm(item_emb)
        if self.is_time:
            time_diff = self.calculate_time_diff(time_stamps)
        else:
            time_diff = None
        for i in range(self.num_layers):
            item_emb, time_diff = self.ssd_layers[i](item_emb, time_diff)
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output


    def calculate_loss(self, item_seq, item_seq_len, labels, time_stamps):
        """
        计算损失
        :param time_stamps:
        :param item_seq:
        :param item_seq_len:
        :param labels:
        :return:
        """
        seq_output = self(item_seq, item_seq_len, time_stamps)
        test_item_emb = self.adapter(self.item_embedding)
        logits = torch.einsum('bh,nh->bn', seq_output, test_item_emb)
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len, time_stamps):
        """
        全排序预测
        :param time_stamps:
        :param item_seq:
        :param item_seq_len:
        :return:
        """
        seq_output = self(item_seq, item_seq_len, time_stamps)
        test_item_emb = self.adapter(self.item_embedding)
        scores = torch.einsum('bh,nh->bn', seq_output, test_item_emb)
        return scores

    def training_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        loss = self.calculate_loss(item_seq=user_item_seq, item_seq_len=length, labels=labels, time_stamps=time_stamps)
        return loss

    def validation_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        scores = self.full_sort_predict(item_seq=user_item_seq, item_seq_len=length, time_stamps=time_stamps)
        sort_index, batch = self.calc_matrix(scores, labels)
        return sort_index, batch

