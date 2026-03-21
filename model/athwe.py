# -*- coding: utf-8 -*-
# @Author : Hao Fan
# @Time : 2026/3/20
import torch
from torch import nn
from torch.nn import functional as F

from model.encoder.transformer import TransformerEncoder
import pickle
import math
import random
from pytorch_wavelets import DWT1D, IDWT1D
from sklearn_extra.cluster import KMedoids
# from sklearn.decomposition import PCA
import numpy as np
from model.abstract import AbstractSRModel


class MOERouter(nn.Module):
    """
    Router for dynamic (routed) experts only.
    """

    def __init__(self, hidden_dim, routed_expert_num, moe_topk):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, routed_expert_num)
        self.routed_expert_num = routed_expert_num
        self.moe_topk = moe_topk

    def forward(self, hidden_states):
        # hidden_states: (B, T, D) or (N, D)
        original_shape = hidden_states.size()
        if len(original_shape) == 3:
            # If input is (B, T, D), reshape to (B*T, D)
            hidden_states = hidden_states.view(-1, original_shape[-1])

        logits = self.gate(hidden_states)  # (N, E_r)
        # Add numerical stability - subtract the max value before softmax
        logits_max = torch.max(logits, dim=-1, keepdim=True)[0].detach()
        logits_stable = logits - logits_max
        probs = F.softmax(logits_stable, dim=-1)  # (N, E_r)
        # Top-k routing
        topk_vals, topk_idx = torch.topk(probs, self.moe_topk, dim=-1)  # (N, K)

        # Add numerical stability
        if self.moe_topk == 1:
            # When moe_topk=1, use 1.0 as weight to avoid division
            norm_vals = torch.ones_like(topk_vals)
        else:
            # Add small epsilon to avoid division by zero, and use clamp to avoid extreme values
            sum_vals = topk_vals.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            norm_vals = topk_vals / sum_vals

        return logits, norm_vals, topk_idx


class SparseMOE(nn.Module):
    """
    Simplified MoE with separate shared and routed expert lists.
    Shared experts always active; routed experts selected by MOERouter.
    """

    def __init__(self, hidden_dim, routed_expert_nums, moe_topk, shared_expert_nums=0, cosine_lambda=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.shared_expert_nums = shared_expert_nums
        self.routed_expert_nums = routed_expert_nums
        self.moe_topk = moe_topk
        self.cosine_lambda = torch.tensor(cosine_lambda)
        self.current_diversity_loss = torch.tensor(0.0)

        # Shared experts: always active
        if shared_expert_nums > 0:
            self.shared_experts = nn.ParameterList([
                nn.Parameter(torch.Tensor(1, hidden_dim), requires_grad=True)
                for _ in range(shared_expert_nums)
            ])
            # Initialize the shared expert parameters
            for param in self.shared_experts:
                nn.init.normal_(param, std=0.1)

        # Routed experts: using Parameter instead of ModuleList
        self.routed_experts = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, hidden_dim), requires_grad=True)
            for _ in range(self.routed_expert_nums)
        ])
        # Initialize the expert parameters
        for param in self.routed_experts:
            nn.init.normal_(param, std=0.1)

        self.router = MOERouter(hidden_dim, self.routed_expert_nums, moe_topk)

    def forward(self, x, routing_features=None, time_embedding=None):
        B, T, D = x.size()
        N = B * T
        hidden = x.view(N, D)

        routing_input = routing_features.view(N, D) if routing_features is not None else hidden

        # 1) Shared experts forward
        if hasattr(self, 'shared_experts') and self.shared_expert_nums > 0:
            shared_outs = []
            for exp in self.shared_experts:
                shared_outs.append(hidden * exp)  # Element-wise multiplication
            shared_sum = torch.stack(shared_outs, dim=0).sum(dim=0)
        else:
            shared_sum = torch.zeros((N, D), device=hidden.device, dtype=hidden.dtype)

        # 2) Routed experts forward & routing
        _, weights, indices = self.router(routing_input)

        # Expert computation using Parameter experts
        routed_outs = []
        for expert in self.routed_experts:
            expert_out = hidden * expert  # Element-wise multiplication
            routed_outs.append(expert_out)

        routed_outs = torch.stack(routed_outs, dim=0)  # [E_r, N, D']

        # Calculate diversity loss
        E, N, D = routed_outs.size()
        flat = routed_outs.reshape(E, -1)
        normalized = F.normalize(flat, p=2, dim=1, eps=1e-8)
        sim_matrix = torch.matmul(normalized, normalized.T)
        sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

        mask = 1.0 - torch.eye(E, device=sim_matrix.device)
        sim_loss = (sim_matrix * mask).sum() / max(E * (E - 1), 1)
        self.current_diversity_loss = sim_loss * self.cosine_lambda.to(sim_loss.device)

        out_dim = routed_outs.size(-1)
        routed_outs = routed_outs.permute(1, 0, 2)

        idx = indices.unsqueeze(-1).expand(-1, self.moe_topk, out_dim)
        sel = routed_outs.gather(1, idx)
        dyn_sum = (sel * weights.unsqueeze(-1)).sum(dim=1)

        combined = shared_sum + dyn_sum
        out = combined.view(B, T, out_dim)

        return out

    def get_expert_diversity_loss(self):
        return self.current_diversity_loss


class AlignWaveletFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        H = int(config["hidden_size"])
        J = config["wavelet_J"] if "wavelet_J" in config else 4
        wave = "db4"
        self.initializer_range = config["initializer_range"]
        self.dwt = DWT1D(J=J, wave=wave)
        self.idwt = IDWT1D(wave=wave)
        self.levels = J
        self.modals = ["item", "txt", "img"]
        self.H = H
        self.low_g = nn.ModuleDict({
            m: nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.modals])
            for m in self.modals
        })
        self.high_g = nn.ModuleDict({
            m: nn.ModuleList([
                nn.ModuleList([nn.Linear(H, H, bias=False) for _ in self.modals])
                for _ in range(J)
            ])
            for m in self.modals
        })
        self.weight = nn.Parameter(torch.tensor(config["initializer_weight"], dtype=torch.float))

    def forward(self, x):
        # x: (B, L, 3H) →
        xs = dict(zip(self.modals, x.split(self.H, dim=-1)))

        lows, highs = {}, {}
        for m in self.modals:
            l, h = self.dwt(xs[m].permute(0, 2, 1))  # → (B, H, L)
            lows[m] = l.permute(0, 2, 1)  # → (B, L, H)
            highs[m] = [h_i.permute(0, 2, 1) for h_i in h]  # list of (B, L, H)
        outs = []

        for idx, m in enumerate(self.modals):
            fused_l = sum(
                torch.sigmoid(self.low_g[m][j](lows[s])) * lows[s]
                for j, s in enumerate(self.modals)
            )
            fused_h = []
            for k in range(self.levels):
                term = sum(
                    torch.sigmoid(self.high_g[m][k][j](highs[s][k])) * highs[s][k]
                    for j, s in enumerate(self.modals)
                )
                fused_h.append(term.permute(0, 2, 1))  # → (B, H, L)
            # (B, L, H)
            fused_m = self.idwt((fused_l.permute(0, 2, 1), fused_h)).permute(0, 2, 1)
            outs.append((fused_m - xs[m]) * self.weight[idx])
        return outs


class TimeMoEFusion(nn.Module):
    """
    Time-aware MoE Fusion module:
    - Input: Original modality features
    - Routing: Concatenated time embeddings
    - Expert: Linear Layer with diversity loss
    """

    def __init__(self, config):
        super(TimeMoEFusion, self).__init__()
        # Basic configurations
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"] if "inner_size" in config else 4 * self.hidden_size
        self.hidden_dropout_prob = config["hidden_dropout_prob"] if "hidden_dropout_prob" in config else 0.1
        self.layer_norm_eps = config["layer_norm_eps"] if "layer_norm_eps" in config else 1e-12

        # Time encoding configurations
        self.a = config["time_span"] if "time_span" in config else 1000
        k_value = config["time_scale"] if "time_scale" in config else 2e7
        self.k = torch.tensor(k_value, dtype=torch.float)  # Convert k to tensor
        self.time_embed = nn.Embedding(self.a + 10, self.hidden_size)

        # Add time routing projection layer
        self.time_routing_proj = nn.Linear(self.hidden_size, self.hidden_size * 3)
        self.routing_activation = nn.ReLU()
        self.routing_dropout = nn.Dropout(self.hidden_dropout_prob)

        # MoE configurations
        self.moe_topk = config["moe_topk"] if "moe_topk" in config else 2
        self.routed_expert_nums = config["routed_expert_nums"] if "routed_expert_nums" in config else 4
        self.shared_expert_nums = config["shared_expert_nums"] if "shared_expert_nums" in config else 1
        self.cosine_lambda = config["cosine_lambda"] if "cosine_lambda" in config else 1.0

        # Single MoE layer with linear experts
        self.moe = SparseMOE(
            hidden_dim=self.hidden_size * 3,  # Input: concatenated [item, txt, img]
            routed_expert_nums=self.routed_expert_nums,
            moe_topk=self.moe_topk,
            shared_expert_nums=self.shared_expert_nums,
            cosine_lambda=self.cosine_lambda
        )

        # Add wavelet alignment module
        self.wavelet_aligner = AlignWaveletFusion(config)

    def compute_temporal_embedding(self, timestamp):
        """Compute temporal embeddings using exponential saturation function"""
        # Calculate time differences
        time_diff = self.calculate_time_diff(timestamp)
        # Move k to the same device as time_diff
        k = self.k.to(time_diff.device)
        # Apply exponential saturation: a * (1 - exp(-x / abs(k)))
        time_values = torch.round(self.a * (1 - torch.exp(-time_diff / torch.abs(k))))
        # Convert to long for embedding lookup
        time_id = time_values.long()
        # Get time embeddings
        time_embed = self.time_embed(time_id)
        return time_embed

    def calculate_time_diff(self, timestamp):
        """Calculate time differences between consecutive timestamps"""
        # Mask for valid timestamps
        valid_mask = (timestamp != 0).float()

        # Compute differences
        diffs = timestamp[:, 1:] - timestamp[:, :-1]
        diffs = torch.where(diffs < 0, torch.zeros_like(diffs), diffs)
        # Prepend ones for first position
        first_col = torch.ones_like(timestamp[:, :1])
        time_diff = torch.cat([first_col, diffs], dim=1)

        # Apply valid mask
        return time_diff * valid_mask

    def forward(self, vector, timestamp):
        """
        Forward pass through the MoE architecture
        Args:
            vector: concatenated input features [B, L, 3H]
            timestamp: timestamp sequence [B, L]
        Returns:
            Processed features split back into three modalities
        """
        B, L, H3 = vector.size()
        H = H3 // 3

        # Split input vector into three modalities and create new tensors
        item_emb, txt_emb, img_emb = [x.clone() for x in vector.split(H, dim=-1)]

        # Get time embeddings for routing
        time_embedding = self.compute_temporal_embedding(timestamp)  # [B, L, H]

        # Apply wavelet alignment for feature enhancement
        modal_residuals = self.wavelet_aligner(vector)

        # Add residual connections without inplace operations
        item_emb = item_emb + modal_residuals[0]
        txt_emb = txt_emb + modal_residuals[1]
        img_emb = img_emb + modal_residuals[2]

        # Create enhanced input features
        enhanced_features = torch.cat([item_emb, txt_emb, img_emb], dim=-1)  # [B, L, 3H]

        # Create routing features using time embeddings through linear projection
        routing_features = self.time_routing_proj(time_embedding)  # [B, L, 3H]
        routing_features = self.routing_activation(routing_features)
        routing_features = self.routing_dropout(routing_features)

        # Forward through MoE
        output = self.moe(enhanced_features, routing_features=routing_features)  # [B, L, 3H]

        # Split output back into three modalities
        item_out = output[:, :, :H]
        txt_out = output[:, :, H:2 * H]
        img_out = output[:, :, 2 * H:]

        return item_out, txt_out, img_out


class ATHWE(AbstractSRModel):
    def __init__(self, config):
        super(ATHWE, self).__init__(config)

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
        self.diversity_weight = config["diversity_weight"] if "diversity_weight" in config else 1.0

        self.fusion_weights = nn.Parameter(torch.ones(3))

        self.n_clusters = config["n_clusters"] if "n_clusters" in config else 20
        # self.n_components = config["n_components"] if "n_components" in config else 64

        self.item_embedding = nn.Embedding(self.item_num + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.seq_len, self.hidden_size)
        self.item_seq = TransformerEncoder(
            n_layers=self.n_layers, n_heads=self.n_heads,
            hidden_size=self.hidden_size, inner_size=self.inner_size, hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
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

        # ---
        self.item_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.txt_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.img_ln = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        self.txt_classifier = nn.Linear(self.hidden_size, self.n_clusters)
        self.img_classifier = nn.Linear(self.hidden_size, self.n_clusters)

        self.cluster_loss_fct = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        self.time_moe = TimeMoEFusion(config)
        self.placeholder_txt = nn.Linear(self.hidden_size, self.hidden_size)
        self.placeholder_img = nn.Linear(self.hidden_size, self.hidden_size)

        self.apply(self._init_weights)

        img_feat,text_feat=nn.Embedding.from_pretrained(torch.load(config['vision_embed_path']),freeze=True),\
                           nn.Embedding.from_pretrained(torch.load(config['text_embed_path']),freeze=True)
        self.img_embedding=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,img_feat.weight.shape[-1]),img_feat.weight),dim=0
        ),freeze=True)
        self.txt_embedding=nn.Embedding.from_pretrained(torch.cat(
            (torch.zeros(1,text_feat.weight.shape[-1]),text_feat.weight),dim=0
        ),freeze=True)

        self.txt_projection = nn.Linear(self.txt_embedding.weight.shape[-1], self.hidden_size)
        self.img_projection = nn.Linear(self.img_embedding.weight.shape[-1], self.hidden_size)


        cat_emb = torch.load(config['cat_path']).float()  # 从config读取完整路径 ../processed/Scientific/cat.pt
        self.cat_embedding = nn.Embedding.from_pretrained(cat_emb)
        self.cat_linear = nn.Linear(3 * self.hidden_size, cat_emb.shape[-1])
        self.cat_criterion = nn.BCEWithLogitsLoss()

        print("Performing ZCA whitening and clustering on text and image embeddings...")

        txt_embeddings = self.txt_embedding.weight.detach().cpu().numpy()
        txt_embeddings_whitened = self.zca_whitening(txt_embeddings)
        print(f"Text embedding ZCA whitening completed; dimensionality remains {txt_embeddings.shape[1]}")

        txt_kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            random_state=config["seed"],
            method='alternate',
            init='k-medoids++',
            max_iter=100
        )
        self.txt_clusters = torch.tensor(txt_kmedoids.fit_predict(txt_embeddings_whitened))
        print(f"Text clustering completed; {self.n_clusters} clusters generated")

        img_embeddings = self.img_embedding.weight.detach().cpu().numpy()
        img_embeddings_whitened = self.zca_whitening(img_embeddings)
        print(f"Image embedding ZCA whitening completed; dimensionality remains {img_embeddings.shape[1]}")

        img_kmedoids = KMedoids(
            n_clusters=self.n_clusters,
            random_state=config["seed"],
            method='alternate',
            init='k-medoids++',
            max_iter=100
        )
        self.img_clusters = torch.tensor(img_kmedoids.fit_predict(img_embeddings_whitened))
        print(f"Image clustering completed; {self.n_clusters} clusters generated")

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def zca_whitening(self, embeddings, epsilon=1e-5):

        if isinstance(embeddings, np.ndarray):
            E = torch.from_numpy(embeddings).float()
        else:
            E = embeddings.float()

        N, D = E.shape
        E_mean = E.mean(dim=0, keepdim=True)
        E_centered = E - E_mean
        Sigma = torch.matmul(E_centered.T, E_centered) / N

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
            eigenvalues = torch.clamp(eigenvalues, min=epsilon)
        except:
            eigenvalues, eigenvectors = torch.symeig(Sigma, eigenvectors=True)
            eigenvalues = torch.clamp(eigenvalues, min=epsilon)

        eigenvalues_sqrt_inv = eigenvalues.rsqrt()

        W = torch.matmul(
            torch.matmul(eigenvectors, torch.diag(eigenvalues_sqrt_inv)),
            eigenvectors.T
        )

        E_whitened = torch.matmul(E_centered, W)

        return E_whitened.detach().cpu().numpy()

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

        item_emb, txt_emb, img_emb = self.time_moe(torch.cat([item_emb, txt_emb, img_emb], dim=-1), timestamp)

        item_emb_o = self.dropout(self.item_ln(item_emb))
        txt_emb_o = self.dropout(self.txt_ln(txt_emb))
        img_emb_o = self.dropout(self.img_ln(img_emb))

        extended_attention_mask = self.get_attention_mask(item_seq)
        item_seq_full = self.item_seq(item_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        txt_seq_full = self.txt_seq(txt_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        img_seq_full = self.img_seq(img_emb_o, extended_attention_mask, output_all_encoded_layers=True)[-1]
        item_seq = self.gather_indexes(item_seq_full, item_seq_len - 1)
        txt_seq = self.gather_indexes(txt_seq_full, item_seq_len - 1)
        img_seq = self.gather_indexes(img_seq_full, item_seq_len - 1)

        item_emb_full = self.item_embedding.weight
        txt_emb_full = self.txt_projection(self.txt_embedding.weight)
        img_emb_full = self.img_projection(self.img_embedding.weight)

        item_score = torch.matmul(item_seq, item_emb_full.transpose(0, 1))
        txt_score = torch.matmul(txt_seq, txt_emb_full.transpose(0, 1))
        img_score = torch.matmul(img_seq, img_emb_full.transpose(0, 1))

        normalized_weights = F.softmax(self.fusion_weights, dim=0)

        score = (normalized_weights[0] * item_score +
                 normalized_weights[1] * txt_score +
                 normalized_weights[2] * img_score)
        return [item_emb, txt_emb, img_emb], [item_seq, txt_seq, img_seq], score

    def calculate_loss(self, item_seq, item_seq_len, labels, time_stamps):
        item_emb_seq, seq_vectors, score = self.forward(item_seq, item_seq_len, time_stamps)
        loss = self.loss_fct(score, labels)

        # Add MoE diversity loss with weight from config
        moe_diversity_loss = self.time_moe.moe.get_expert_diversity_loss() * self.diversity_weight

        total_loss = loss + self.IDCL(seq_vectors[0], labels) + \
                    self.supervised_attribute_loss(item_seq) + \
                    self.unsupervised_cluster_loss(item_seq) + \
                    self.TCL(item_seq, item_seq_len, time_stamps, labels, item_emb_seq, seq_vectors) + \
                    moe_diversity_loss

        return total_loss

        # return loss + self.IDCL(seq_vectors[0], labels) + self.CP(item_seq) + self.PCL(item_seq, item_seq_len, time_stamps, labels, item_emb_seq, seq_vectors)

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
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),
                                 neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def supervised_attribute_loss(self, input_idx, padding_idx=0):
        item_list = input_idx.flatten()
        # 找到非填充值的索引
        non_pad_mask = (input_idx != padding_idx)

        # 检查是否有非填充值
        if not non_pad_mask.any():
            # 如果没有非填充值，返回0损失
            return torch.tensor(0.0, device=input_idx.device, requires_grad=True)

        # 获取非填充值对应的展平索引
        nonzero_idx = torch.where(non_pad_mask.flatten())[0]

        item_emb = self.item_embedding(item_list)
        txt_emb = self.txt_projection(self.txt_embedding(item_list))
        img_emb = self.img_projection(self.img_embedding(item_list))

        item_attribute_score = self.cat_linear(torch.cat([item_emb, txt_emb, img_emb], dim=-1))

        item_attribute_target = self.cat_embedding(item_list)

        attr_loss = self.cat_criterion(item_attribute_score[nonzero_idx], item_attribute_target[nonzero_idx])
        return attr_loss

    def seq2seq_contrastive(self, seq_1, seq_2, same_pos_id):
        seq_1 = F.normalize(seq_1, dim=1)
        seq_2 = F.normalize(seq_2, dim=1)

        pos_logits = (seq_1 * seq_2).sum(dim=1) / self.phcl_temperature
        pos_logits = torch.exp(pos_logits)
        neg_logits = torch.matmul(seq_1, seq_2.transpose(0, 1)) / self.phcl_temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),
                                 neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean() * self.phcl_weight

    def TCL(self, item_seq, item_seq_len, timestamp, labels, item_emb_seq, seq_vectors):
        beta = self.beta
        item_seq = item_seq
        item_seq_len = item_seq_len
        timestamp = timestamp

        mask_rates = beta * torch.ones_like(item_seq_len, dtype=torch.float)
        num_masks = torch.floor(mask_rates).long()
        batch_size, seq_len = item_seq.shape
        mask_matrix = torch.ones(batch_size, seq_len, dtype=torch.bool, device=item_seq.device)

        for i in range(batch_size):
            if num_masks[i] > 0:
                mask_indices = torch.randperm(item_seq_len[i])[:num_masks[i]]
                mask_matrix[i, mask_indices] = False

        item_seq_aug = torch.where(mask_matrix, item_seq, torch.tensor(-1, device=item_seq.device))
        time_embedding = self.time_moe.compute_temporal_embedding(timestamp)
        txt_embs_orig, img_embs_orig = item_emb_seq[1], item_emb_seq[2]

        def create_temporal_augmentation(original_embs, placeholder_layer, time_embed, mask_matrix):
            embs_aug = original_embs * mask_matrix.unsqueeze(-1).float()
            time_aware_placeholder = placeholder_layer(time_embed)
            time_aware_placeholder = time_aware_placeholder * (~mask_matrix).unsqueeze(-1).float()
            return embs_aug + time_aware_placeholder

        txt_embs_aug = create_temporal_augmentation(txt_embs_orig, self.placeholder_txt, time_embedding, mask_matrix)
        img_embs_aug = create_temporal_augmentation(img_embs_orig, self.placeholder_img, time_embedding, mask_matrix)

        modal_embs_aug = [txt_embs_aug, img_embs_aug]
        modal_encoders = [self.txt_seq, self.img_seq]
        modal_norms = [self.txt_ln, self.img_ln]

        augmented_sequences = []
        for embs, encoder, norm_layer in zip(modal_embs_aug, modal_encoders, modal_norms):
            processed_embs = self.dropout(norm_layer(embs))
            attention_mask = self.get_attention_mask(item_seq)
            full_seq = encoder(processed_embs, attention_mask, output_all_encoded_layers=True)[-1]
            seq_output = self.gather_indexes(full_seq, item_seq_len - 1)
            augmented_sequences.append(seq_output)

        txt_seq_aug, img_seq_aug = augmented_sequences
        pos_id = labels
        batch_size = pos_id.shape[0]
        pos_mask = pos_id.unsqueeze(1) == pos_id.unsqueeze(0)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=pos_id.device)
        contrast_mask = torch.logical_xor(pos_mask, self_mask)

        txt_loss = self.seq2seq_contrastive(seq_vectors[1], txt_seq_aug, contrast_mask)
        img_loss = self.seq2seq_contrastive(seq_vectors[2], img_seq_aug, contrast_mask)

        return 0.5 * (txt_loss + img_loss)

    def unsupervised_cluster_loss(self, input_idx, padding_idx=0):

        item_list = input_idx.flatten()
        # 找到非填充值的索引
        non_pad_mask = (input_idx != padding_idx)

        # 检查是否有非填充值
        if not non_pad_mask.any():
            # 如果没有非填充值，返回0损失
            return torch.tensor(0.0, device=input_idx.device, requires_grad=True)

        # 获取非填充值对应的展平索引
        nonzero_idx = torch.where(non_pad_mask.flatten())[0]

        txt_emb = self.txt_projection(self.txt_embedding(item_list))
        img_emb = self.img_projection(self.img_embedding(item_list))

        txt_clusters = self.txt_clusters.to(item_list.device)
        img_clusters = self.img_clusters.to(item_list.device)

        txt_labels = txt_clusters[item_list]
        img_labels = img_clusters[item_list]

        txt_pred = self.txt_classifier(txt_emb)
        txt_loss = self.cluster_loss_fct(txt_pred[nonzero_idx], txt_labels[nonzero_idx])

        img_pred = self.img_classifier(img_emb)
        img_loss = self.cluster_loss_fct(img_pred[nonzero_idx], img_labels[nonzero_idx])

        return (txt_loss + img_loss) / 2