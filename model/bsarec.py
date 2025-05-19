# -*- coding: utf-8 -*-
# @Author : Anonymous
# @Time : 2025/3/5

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .encoder.transformer import TransformerEncoder
    from .abstract import AbstractSRModel
except:
    from encoder.transformer import TransformerEncoder
    from abstract import AbstractSRModel

class BSARec(AbstractSRModel):
    def __init__(self, config):
        super(BSARec, self).__init__(config)

        # load parameters info
        self.hidden_size = config["hidden_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.c = config["c"]
        self.alpha = config["alpha"]
        self.num_hidden_layers = config["n_layers"]
        self.num_attention_heads = config["n_heads"]
        self.attention_probs_dropout_prob = config["attn_dropout_prob"]
        self.initializer_range = config['initializer_range']

        self.item_embedding = nn.Embedding(
            self.item_num + 1, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.seq_len, self.hidden_size)

        self.trm_encoder = BSARecEncoder(self.hidden_size, self.hidden_dropout_prob, self.hidden_act, self.c, self.alpha, self.num_hidden_layers, self.num_attention_heads, self.attention_probs_dropout_prob)

        self.LayerNorm = LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.loss_fct = nn.CrossEntropyLoss()

        self.apply(self._init_weights)

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding

        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def calculate_loss(self, item_seq, item_seq_len, labels):
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, labels)
        return loss

    def full_sort_predict(self, item_seq, item_seq_len):
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
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

class BSARecEncoder(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, hidden_act, c, alpha, num_hidden_layers, num_attention_heads, attention_probs_dropout_prob):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(hidden_size, hidden_dropout_prob, hidden_act, c, alpha, num_attention_heads, attention_probs_dropout_prob)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, hidden_act, c, alpha, num_attention_heads, attention_probs_dropout_prob):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(hidden_size, hidden_dropout_prob, c, alpha, num_attention_heads, attention_probs_dropout_prob)
        self.feed_forward = FeedForward(hidden_size, hidden_act, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, c, alpha, num_attention_heads, attention_probs_dropout_prob):
        super(BSARecLayer, self).__init__()
        self.filter_layer = FrequencyLayer(hidden_size, hidden_dropout_prob, c)
        self.attention_layer = MultiHeadAttention(hidden_size, num_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob)
        self.alpha = alpha

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + (1 - self.alpha) * gsp

        return hidden_states


class FrequencyLayer(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob, c):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.c = c // 2 + 1
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta ** 2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FeedForward(nn.Module):
    def __init__(self, hidden_size, hidden_act, hidden_dropout_prob):
        super(FeedForward, self).__init__()

        hidden_size = hidden_size
        inner_size = 4 * hidden_size

        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


#######################
## Basic Transformer ##
#######################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob, attention_probs_dropout_prob):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states