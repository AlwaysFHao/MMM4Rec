# -*- coding: utf-8 -*-
# @Author : Hao Fan
# @Time : 2026/3/20
import pickle
import random

import torch
from torch import nn
import torch.nn.functional as F

from model.abstract import AbstractSRModel
from model.encoder.transformer import TransformerEncoder


class HM4SR(AbstractSRModel):
    def __init__(self, config):
        super(HM4SR, self).__init__(config)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = float(config["layer_norm_eps"])
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.temperature = config["temperature"]
        self.phcl_temperature = config["phcl_temperature"]
        self.phcl_weight = config["phcl_weight"]
        self.beta = config["beta"]

        self.item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        self.item_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob, attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.txt_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)
        self.img_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act, layer_norm_eps=self.layer_norm_eps)

        self.item_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.txt_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.img_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        img_feat,text_feat=nn.Embedding.from_pretrained(torch.load(config['vision_embed_path']),freeze=True),\
                           nn.Embedding.from_pretrained(torch.load(config['text_embed_path']),freeze=True)
        self.img_embedding=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,img_feat.weight.shape[-1]),img_feat.weight),dim=0
        ),freeze=True)
        self.txt_embedding=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,text_feat.weight.shape[-1]),text_feat.weight),dim=0
        ),freeze=True)

        # 增加模态映射
        self.txt_projection = nn.Linear(self.txt_embedding.weight.shape[-1], self.hidden_size)
        self.img_projection = nn.Linear(self.img_embedding.weight.shape[-1], self.hidden_size)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        # 增加时序信息
        self.time_moe = Temporal_MoE_C(config)

        self.apply(self._init_weights)

        # 增加属性类别预测任务
        cat_emb = torch.load(config['cat_path']).float()  # 从config读取完整路径 ../processed/Scientific/cat.pt
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb)
        self.cat_linear = nn.Linear(3 * self.hidden_size, cat_emb.shape[-1])
        self.cat_criterion = nn.BCEWithLogitsLoss()
        # 增加初始MoE
        self.start_moe = Align_MoE(config)
        # 增加placeholder编码器
        self.placeholder_txt = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.placeholder_img = nn.Linear(2*self.hidden_size, self.hidden_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, timestamp=None):
        item_emb = self.item_embedding(item_seq)
        img_emb=self.img_embedding(item_seq)
        txt_emb=self.txt_embedding(item_seq)

        img_emb=self.img_projection(img_emb)
        txt_emb=self.txt_projection(txt_emb)

        id_pos_emb = self.position_embedding.weight[:item_seq.shape[1]]
        id_pos_emb = id_pos_emb.unsqueeze(0).repeat(item_emb.shape[0], 1, 1)

        item_emb += id_pos_emb
        img_emb += id_pos_emb
        txt_emb += id_pos_emb

        ### 添加MoE ###
        align_info = self.start_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1))
        item_emb += align_info[0]
        txt_emb += align_info[1]
        img_emb += align_info[2]
        ### 添加时序MoE ###
        item_emb, txt_emb, img_emb = self.time_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1), timestamp)
        # 层正则化+dropout
        item_emb_o = self.dropout(self.item_ln(item_emb))
        txt_emb_o = self.dropout(self.txt_ln(txt_emb))
        img_emb_o = self.dropout(self.img_ln(img_emb))
        # 序列编码
        extended_attention_mask = self.get_attention_mask(item_seq)
        item_seq_full = self.item_seq(item_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        txt_seq_full = self.txt_seq(txt_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        img_seq_full = self.img_seq(img_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        item_seq = self.gather_indexes(item_seq_full, item_seq_len - 1)
        txt_seq = self.gather_indexes(txt_seq_full, item_seq_len - 1)
        img_seq = self.gather_indexes(img_seq_full, item_seq_len - 1)
        # 预测
        item_emb_full = self.item_embedding.weight
        txt_emb_full = self.txt_projection(self.txt_embedding.weight)
        img_emb_full = self.img_projection(self.img_embedding.weight)

        item_score = torch.matmul(item_seq, item_emb_full.transpose(0, 1))
        txt_score = torch.matmul(txt_seq, txt_emb_full.transpose(0, 1))
        img_score = torch.matmul(img_seq, img_emb_full.transpose(0, 1))
        score = item_score + txt_score + img_score
        return [item_emb, txt_emb, img_emb], [item_seq, txt_seq, img_seq], score

    def calculate_loss(self, item_seq, item_seq_len, labels, time_stamps):
        item_emb_seq, seq_vectors, score = self.forward(item_seq, item_seq_len, time_stamps)
        loss = self.loss_fct(score, labels)
        return loss + self.IDCL(seq_vectors[0], labels) + self.CP(item_seq) + self.PCL(item_seq, item_seq_len, time_stamps, labels, item_emb_seq, seq_vectors)

    def full_sort_predict(self, item_seq, item_seq_len, time_stamps):
        _, _, score = self.forward(item_seq, item_seq_len, time_stamps)
        return score

    def training_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        loss = self.calculate_loss(item_seq=user_item_seq, item_seq_len=length, labels=labels, time_stamps=time_stamps)
        return loss

    def validation_step(self, batch):
        user, user_item_seq, labels, length, padding_mask, time_stamps = batch
        scores = self.full_sort_predict(item_seq=user_item_seq, item_seq_len=length, time_stamps=time_stamps)
        sort_index, batch = self.calc_matrix(scores, labels)
        return sort_index, batch

    def IDCL(self, seq_pre, labels):
        # from UniSRec
        seq_output = F.normalize(seq_pre, dim=1)
        pos_id = labels
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        pos_items_emb = self.item_embedding(pos_id)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def CP(self, input_idx, padding_idx=0):
        # 展平输入
        item_list = input_idx.flatten()

        # 找到非填充值的索引
        non_pad_mask = (input_idx != padding_idx)

        # 检查是否有非填充值
        if not non_pad_mask.any():
            # 如果没有非填充值，返回0损失
            return torch.tensor(0.0, device=input_idx.device, requires_grad=True)

        # 获取非填充值对应的展平索引
        nonzero_idx = torch.where(non_pad_mask.flatten())[0]

        # 嵌入映射
        item_emb = self.item_embedding(item_list)
        txt_emb = self.txt_projection(self.txt_embedding(item_list))
        img_emb = self.img_projection(self.img_embedding(item_list))

        # 预测类别
        item_attribute_score = self.cat_linear(torch.cat([item_emb, txt_emb, img_emb], dim=-1))

        # 获取答案类别
        item_attribute_target = self.cat_embedding(item_list)

        # 计算损失
        attr_loss = self.cat_criterion(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss

    def seq2seq_contrastive(self, seq_1, seq_2, same_pos_id):
        seq_1 = F.normalize(seq_1, dim=1)
        seq_2 = F.normalize(seq_2, dim=1)

        pos_logits = (seq_1 * seq_2).sum(dim=1) / self.phcl_temperature
        pos_logits = torch.exp(pos_logits)
        neg_logits = torch.matmul(seq_1, seq_2.transpose(0, 1)) / self.phcl_temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean() * self.phcl_weight

    def PCL(self, item_seq, item_seq_len, timestamp, labels, item_emb_seq, seq_embs):
        beta = self.beta
        # 增强部分
        num_mask = torch.floor(item_seq_len * beta).long().tolist()
        masked_item_seq = item_seq.cpu().detach().numpy().copy()
        for i in range(item_seq.shape[0]):
            mask_index = random.sample(range(item_seq_len[i]), k=num_mask[i])
            masked_item_seq[i, mask_index] = -1
        item_seq_aug = torch.tensor(masked_item_seq, dtype=torch.long, device=item_seq.device)
        # 占位符替换物品
        id_embs, txt_embs, img_embs = item_emb_seq[0], item_emb_seq[1], item_emb_seq[2]
        time_embedding = self.time_moe.get_time_embedding(timestamp)
        placeholder_mask = (item_seq_aug == -1).unsqueeze(2)
        txt_embs_aug = txt_embs.masked_fill(placeholder_mask, 0.0)
        txt_placeholder = self.placeholder_txt(time_embedding).masked_fill(~placeholder_mask, 0.0)
        txt_embs_aug += txt_placeholder
        img_embs_aug = img_embs.masked_fill(placeholder_mask, 0.0)
        img_placeholder = self.placeholder_img(time_embedding).masked_fill(~placeholder_mask, 0.0)
        img_embs_aug += img_placeholder
        # 增强表征计算
        txt_embs_aug = self.dropout(self.txt_ln(txt_embs_aug))
        img_embs_aug = self.dropout(self.img_ln(img_embs_aug))
        extended_attention_mask = self.get_attention_mask(item_seq)
        txt_seq_full = self.txt_seq(txt_embs_aug, extended_attention_mask, output_all_encoded_layers=True)[-1]
        img_seq_full = self.img_seq(img_embs_aug, extended_attention_mask, output_all_encoded_layers=True)[-1]
        txt_seq = self.gather_indexes(txt_seq_full, item_seq_len - 1)
        img_seq = self.gather_indexes(img_seq_full, item_seq_len - 1)
        # 对比学习计算
        pos_id = labels
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(item_seq.shape[0], dtype=torch.bool, device=item_seq.device))
        txt_loss, img_loss = self.seq2seq_contrastive(seq_embs[1], txt_seq, same_pos_id), self.seq2seq_contrastive(seq_embs[2], img_seq, same_pos_id)
        return (txt_loss + img_loss) / 2

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

class Align_MoE(nn.Module):
    def __init__(self, config):
        super(Align_MoE, self).__init__()
        self.expert_num = config["start_expert_num"]
        self.hidden_size = int(config["hidden_size"])
        self.gate_selection = config["start_gate_selection"]
        self.gate_txt = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_img = nn.Linear(self.hidden_size, self.expert_num)
        self.gate_id = nn.Linear(self.hidden_size, self.expert_num)
        self.expert = nn.ModuleList([nn.Linear(self.hidden_size * 3, self.hidden_size * 3) for _ in range(self.expert_num)])  # 先实现最简单的专家网络
        self.weight = nn.Parameter(torch.tensor(config["initializer_weight"]), requires_grad=True)

    def forward(self, vector):
        # 先只实现softmax
        output = None
        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append(self.expert[i](vector).unsqueeze(2))
            expert_output = torch.cat(expert_output, dim=2)
            output = []
            output.append(self.weight[0] * torch.sum(expert_output[:,:,:,:self.hidden_size] * F.softmax(self.gate_id(vector[:,:,:self.hidden_size]), dim=-1).unsqueeze(3), dim=2))
            output.append(self.weight[1] * torch.sum(expert_output[:,:,:, self.hidden_size:2 * self.hidden_size] * F.softmax(self.gate_txt(vector[:,:,self.hidden_size:2 * self.hidden_size]), dim=-1).unsqueeze(3), dim=2))
            output.append(self.weight[2] * torch.sum(expert_output[:,:,:,2 * self.hidden_size:] * F.softmax(self.gate_img(vector[:,:,2 * self.hidden_size:]), dim=-1).unsqueeze(3), dim=2))
        return output


class Temporal_MoE_C(nn.Module):
    def __init__(self, config):
        super(Temporal_MoE_C, self).__init__()
        self.config = config
        self.interval_scale = config["interval_scale"]
        self.hidden_size = int(config["hidden_size"])
        self.expert_num = config["temporal_expert_num"]
        self.gate_selection = config["temporal_gate_selection"]
        self.gate = nn.Linear(2 * self.hidden_size, self.expert_num)
        self.absolute_w = nn.Linear(1, self.hidden_size)
        self.absolute_m = nn.Linear(self.hidden_size, self.hidden_size)
        self.time_embedding = nn.Embedding(int(self.interval_scale * self.get_interval_num()) + 1, self.hidden_size)

        # 使用 nn.ParameterList 来包装参数
        self.expert = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, self.hidden_size * 3))
            for _ in range(self.expert_num)
        ])

        # 初始化参数
        for param in self.expert:
            nn.init.normal_(param, std=0.1)

    def get_interval_num(self):
        interval_path = self.config['interval_path']  # 完整路径 ../processed/Scientific/max_interval.bin
        with open(interval_path, 'rb') as f:
            return pickle.load(f)

    def get_minmax_day(self):
        daterange_path = self.config['daterange_path']  # 完整路径 ../processed/Scientific/date_range.bin
        with open(daterange_path, 'rb') as f:
            return pickle.load(f)

    def get_time_embedding(self, timestamp):
        timestamp = timestamp.float()
        absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((timestamp.shape[0], 1)).long().to(timestamp.device)
        interval = torch.log2(timestamp[:, 1:] - timestamp[:, :-1] + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        return torch.cat([interval_embedding, absolute_embedding], dim=-1)

    def freq_enhance_ab(self, timestamp):
        freq = 10000
        freq_seq = torch.arange(0, self.hidden_size, 1.0, dtype=torch.float).to(timestamp.device)
        inv_freq = 1 / torch.pow(freq, (freq_seq / self.hidden_size)).view(1, -1) # shape = (64)
        return timestamp * inv_freq

    def forward(self, vector, timestamp):
        # 先只实现softmax
        expert_proba = None
        # 将 timestamp 转换为浮点类型
        timestamp = timestamp.float()
        absolute_embedding = torch.cos(self.freq_enhance_ab(self.absolute_w(timestamp.unsqueeze(2))))
        interval_first = torch.zeros((vector.shape[0], 1)).long().to(timestamp.device)
        interval = torch.log2(timestamp[:, 1:] - timestamp[:, :-1] + 1)
        interval_index = torch.floor(self.interval_scale * interval).long()
        interval_index = torch.cat([interval_first, interval_index], dim=-1)
        interval_embedding = self.time_embedding(interval_index)
        route = F.softmax(self.gate(torch.cat([interval_embedding, absolute_embedding], dim=-1)), dim=-1)
        if self.gate_selection == 'softmax':
            expert_output = []
            for i in range(self.expert_num):
                expert_output.append((vector * self.expert[i]).unsqueeze(2))
            expert_output = torch.cat(expert_output, dim=2)
            expert_proba = torch.sum(expert_output * route.unsqueeze(3), dim=2)
        return expert_proba[:, :, :self.hidden_size], expert_proba[:, :, self.hidden_size: 2 * self.hidden_size], expert_proba[:, :, 2 * self.hidden_size:]