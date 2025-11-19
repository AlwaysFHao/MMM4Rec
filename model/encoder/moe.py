# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2025/1/11
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn.functional import scaled_dot_product_attention

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 计算门控网络的输出
        gate_outputs = F.softmax(self.gating_network(x), dim=1)  # 计算每个专家的权重

        # 每个专家的输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts],
                                     dim=-2)  # shape: (batch_size, num_experts, output_dim)

        # 对专家输出加权求和
        # output = gate_outputs.transpose(-1, -2).unsqueeze(-1).contiguous() * expert_outputs
        # output = torch.sum(output, dim=1)
        # output = torch.einsum('bse,bsej->bsj', gate_outputs, expert_outputs)  # shape: (batch_size, output_dim)
        output = torch.sum(gate_outputs.unsqueeze(-1) * expert_outputs, dim=-2)
        return output
