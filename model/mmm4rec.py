# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/19

import torch
import torch.nn as nn
from einops import repeat

try:
    from .abstract import AbstractSRModel
    from .encoder.ssdv2 import TiSSDLayer, CoTiSSDLayer
    from .encoder.moe import MoE
except:
    from encoder.ssdv2 import TiSSDLayer
    from abstract import AbstractSRModel
    from encoder.moe import MoE


class MMM4Rec(AbstractSRModel):
    def __init__(self, config):
        super(MMM4Rec, self).__init__(config)
        # Hyperparameters for TiM4Rec
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout_prob = config['dropout_prob']
        self.embedding_drop_prob = config['embedding_drop_prob']
        self.time_dropout_prob = config['time_dropout_prob']

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
        self.inner_size = config['inner_size']
        self.n_groups = config['n_groups']

        if 'is_modality_bias' in config.keys():
            self.is_modality_bias = config['is_modality_bias']
        else:
            self.is_modality_bias = True

        # self.num_expert = config['num_expert']
        if 'temperature' in config.keys():
            self.temperature = float(config['temperature'])
        else:
            self.temperature = 1.0

        if 'in_batch_loss' in config.keys():
            self.in_batch_loss = config['in_batch_loss']
        else:
            self.in_batch_loss = False

        assert (self.hidden_size * self.expand) % self.head_dim == 0, \
            f'hidden_size * expand {self.hidden_size * self.expand} can\'t divisible by head_dim {self.head_dim} !'
        self.n_heads = (self.hidden_size * self.expand) // self.head_dim

        self.vision_in_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        self.text_in_layer_norm = nn.LayerNorm(self.hidden_size, eps=self.norm_eps)
        if self.is_time:
            # self.time_start_token = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.layer_norm_time = nn.LayerNorm(self.max_seq_length, eps=self.norm_eps)
        self.embedding_dropout = nn.Dropout(self.embedding_drop_prob)
        self.time_dropout = nn.Dropout(self.time_dropout_prob)

        text_feature_dim = self.get_text_embedding(config['text_embed_path'])
        if self.is_modality_bias:
            self.text_item_embedding_bias = nn.Parameter(torch.zeros(self.item_num + 1, self.hidden_size), requires_grad=True)
            # self.text_item_embedding_bias = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.text_adapter = nn.Sequential(
            # MoE(text_feature_dim, self.hidden_size, self.num_expert),
            nn.Linear(text_feature_dim, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )

        vision_feature_dim = self.get_vision_embedding(config['vision_embed_path'])
        if self.is_modality_bias:
            self.vision_item_embedding_bias = nn.Parameter(torch.zeros(self.item_num + 1, self.hidden_size), requires_grad=True)
            # self.vision_item_embedding_bias = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
            # self.vision_item_embedding_bias = self.text_item_embedding_bias
        self.vision_adapter = nn.Sequential(
            # MoE(vision_feature_dim, self.hidden_size, self.num_expert),
            nn.Linear(vision_feature_dim, self.hidden_size),
            # nn.LayerNorm(self.hidden_size)
        )

        self.encoder_ssd_layers = nn.ModuleList([
            TiSSDLayer(
                d_model=self.hidden_size,
                inner_size=self.inner_size,
                seq_len=self.max_seq_length,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                ngroups=self.n_groups,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                time_drop_out=self.time_dropout_prob,
                is_ffn=self.is_ffn,
                is_time=self.is_time,
                p2p_residual=self.p2p_residual,
                norm_eps=self.norm_eps
            ) for _ in range(self.num_layers)
        ])

        self.decoder_ssd_layers = nn.ModuleList([
            CoTiSSDLayer(
                d_model=self.hidden_size,
                inner_size=self.inner_size,
                seq_len=self.max_seq_length,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                num_layers=self.num_layers,
                head_dim=self.head_dim,
                ngroups=self.n_groups,
                chunk_size=self.chunk_size,
                dropout=self.dropout_prob,
                time_drop_out=self.time_dropout_prob,
                is_ffn=self.is_ffn,
                is_time=self.is_time,
                p2p_residual=self.p2p_residual,
                norm_eps=self.norm_eps,
                ssd_block=self.encoder_ssd_layers[i].ssd
            ) for i in range(self.num_layers)
        ])

        self.item_mm_fusion = None
        if 'item_mm_fusion' in config.keys():
            self.item_mm_fusion = str(config['item_mm_fusion'])

        if self.item_mm_fusion == 'dynamic_shared':
            self.fusion_factor = nn.Parameter(data=torch.tensor([0.0, 0.0], dtype=torch.float))
        elif self.item_mm_fusion == 'dynamic_instance':
            self.fusion_factor = nn.Parameter(data=torch.zeros((self.item_num + 1, 2), dtype=torch.float))
        else:
            self.register_buffer('fusion_factor', torch.tensor(1, dtype=torch.float))

        self.loss_fct = nn.CrossEntropyLoss()
        # tau = 1.0
        # if 'tau' in config.keys():
        #     tau = config['tau']

        self.apply(self._init_weights)

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
        time_diff = self.layer_norm_time(self.time_dropout(time_diff))
        # [batch_size, seq_len] -> [batch_size, n_heads, seq_len]
        time_diff = repeat(time_diff, 'b l -> b h l', h=self.n_heads)
        return time_diff

    def get_text_embedding(self, path):
        text_embed = torch.load(path)
        text_embed = torch.cat([torch.zeros(1, text_embed.shape[-1]), text_embed], dim=0)
        self.register_buffer('text_embedding', text_embed)
        # self.item_embedding = nn.Parameter(text_embed)
        return text_embed.shape[-1]

    def get_vision_embedding(self, path):
        vision_embed = torch.load(path)
        vision_embed = torch.cat([torch.zeros(1, vision_embed.shape[-1]), vision_embed], dim=0)
        self.register_buffer('vision_embedding', vision_embed)
        # self.item_embedding = nn.Parameter(text_embed)
        return vision_embed.shape[-1]

    def forward(self, item_seq, item_seq_len, time_stamp):
        if self.is_modality_bias:
            item_vision_emb = self.vision_adapter(self.vision_embedding[item_seq]) + self.vision_item_embedding_bias[item_seq]
            item_text_emb = self.text_adapter(self.text_embedding[item_seq]) + self.text_item_embedding_bias[item_seq]
            # item_vision_emb = self.vision_adapter(self.vision_embedding[item_seq]) + self.vision_item_embedding_bias(item_seq)
            # item_text_emb = self.text_adapter(self.text_embedding[item_seq]) + self.text_item_embedding_bias(item_seq)
        else:
            item_vision_emb = self.vision_adapter(self.vision_embedding[item_seq])
            item_text_emb = self.text_adapter(self.text_embedding[item_seq])

        # item_vision_emb = nn.functional.normalize(item_vision_emb, dim=1)
        # item_text_emb = nn.functional.normalize(item_text_emb, dim=1)

        item_vision_emb = self.vision_in_layer_norm(self.embedding_dropout(item_vision_emb))
        item_text_emb = self.text_in_layer_norm(self.embedding_dropout(item_text_emb))

        if self.is_time:
            vision_time_diff = self.calculate_time_diff(time_stamp)
            text_time_diff = self.calculate_time_diff(time_stamp)
            # text_time_diff = vision_time_diff
        else:
            vision_time_diff = None
            text_time_diff = None

        for i in range(self.num_layers):
            item_text_emb, text_time_diff = self.encoder_ssd_layers[i](item_text_emb, text_time_diff)
            item_vision_emb, vision_time_diff = self.decoder_ssd_layers[i](encoder_input=item_text_emb, decoder_input=item_vision_emb, encoder_time_diff=text_time_diff, decoder_time_diff=vision_time_diff)

        seq_output = self.gather_indexes(item_vision_emb, item_seq_len - 1)
        return seq_output

    def calculate_score(self, user_emb, labels=None):
        """
        计算分数
        :param user_emb: 用户兴趣表示
        :param labels: 当计算in_batch_loss时，传入的batch内item索引
        :return: user在各个item上的兴趣分数
        """
        if self.in_batch_loss:
            assert labels is not None, 'When calculating in_batch loss, labels cannot be None!'
            if self.is_modality_bias:
                test_vision_item_emb = self.vision_adapter(self.vision_embedding[labels]) + self.vision_item_embedding_bias[labels]
                test_text_item_emb = self.text_adapter(self.text_embedding[labels]) + self.text_item_embedding_bias[labels]
            else:
                test_vision_item_emb = self.vision_adapter(self.vision_embedding[labels])
                test_text_item_emb = self.text_adapter(self.text_embedding[labels])
        else:
            if self.is_modality_bias:
                test_vision_item_emb = self.vision_adapter(self.vision_embedding) + self.vision_item_embedding_bias
                test_text_item_emb = self.text_adapter(self.text_embedding) + self.text_item_embedding_bias
                # test_vision_item_emb = self.vision_adapter(self.vision_embedding) + self.vision_item_embedding_bias.weight
                # test_text_item_emb = self.text_adapter(self.text_embedding) + self.text_item_embedding_bias.weight
            else:
                test_vision_item_emb = self.vision_adapter(self.vision_embedding)
                test_text_item_emb = self.text_adapter(self.text_embedding)


        test_vision_item_emb = nn.functional.normalize(test_vision_item_emb, dim=1)
        test_text_item_emb = nn.functional.normalize(test_text_item_emb, dim=1)

        # test_vision_item_emb = self.vision_in_layer_norm(test_vision_item_emb)
        # test_text_item_emb = self.vision_in_layer_norm(test_text_item_emb)

        text_logits = torch.einsum('bh,nh->bn', user_emb, test_text_item_emb)
        vision_logits = torch.einsum('bh,nh->bn', user_emb, test_vision_item_emb)

        # [b, n, 2]
        modality_logits = torch.stack([text_logits, vision_logits], dim=-1)

        if self.item_mm_fusion in ['dynamic_shared', 'dynamic_instance']:
            if self.in_batch_loss and self.item_mm_fusion == 'dynamic_instance':
                # scores = (modality_logits * torch.softmax(modality_logits * self.fusion_factor[labels], dim=-1)).sum(dim=-1)
                scores = (modality_logits * torch.softmax(self.fusion_factor[labels], dim=-1)).sum(dim=-1)
            else:
                # scores = (modality_logits * torch.softmax(modality_logits * self.fusion_factor, dim=-1)).sum(dim=-1)
                scores = (modality_logits * torch.softmax(self.fusion_factor, dim=-1)).sum(dim=-1)
        else: # 'static'
            scores = modality_logits.sum(dim=-1)
        return scores / self.temperature

    def calculate_loss(self, item_seq, item_seq_len, labels, time_stamps, return_scores=False):
        """
        计算损失
        :param item_seq: 用户的item交互序列
        :param item_seq_len: 交互序列长度
        :param labels: 正样本标签
        :param return_scores: 是否返回训练分数
        :param time_stamps: 时间戳
        :return: 损失 or (损失和分数)
        """
        seq_output = self(item_seq, item_seq_len, time_stamps)
        logits = self.calculate_score(seq_output, labels)
        if self.in_batch_loss:
            batch_size = seq_output.shape[0]
            device = seq_output.device
            batch_labels = torch.arange(batch_size, device=device, dtype=torch.long)
            loss = self.loss_fct(logits, batch_labels)
        else:
            loss = self.loss_fct(logits, labels)
            if return_scores:
                return loss, logits

        return loss

    def full_sort_predict(self, item_seq, item_seq_len, time_stamps):
        """
        全排序预测
        :param item_seq:
        :param item_seq_len:
        :param time_stamps:
        :return:
        """
        seq_output = self(item_seq, item_seq_len, time_stamps)
        scores = self.calculate_score(seq_output)
        return scores
        # return self.mask_visited_items_efficient(item_seq, scores)

    def training_step(self, batch, return_scores=False):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        if return_scores:
            assert self.in_batch_loss is False, 'When calculating in_batch loss, score cannot be returned!'
            loss, scores = self.calculate_loss(item_seq=user_item_seq, item_seq_len=length, labels=labels,
                                       time_stamps=time_stamps, return_scores=return_scores)
            sort_index, batch = self.calc_matrix(scores, labels)
            return loss, sort_index, batch
        else:
            loss = self.calculate_loss(item_seq=user_item_seq, item_seq_len=length, labels=labels,
                                       time_stamps=time_stamps, return_scores=return_scores)
            return loss

    def validation_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        scores = self.full_sort_predict(item_seq=user_item_seq, item_seq_len=length, time_stamps=time_stamps)
        sort_index, batch = self.calc_matrix(scores, labels)
        return sort_index, batch
