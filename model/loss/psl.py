# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/1/2
import torch
import torch.nn as nn


class PSL(nn.Module):
    def __init__(self, activation, tau: float = 1.0):
        super(PSL, self).__init__()
        self.tau = tau
        if activation == 'tanh':
            self.sigma = lambda x: torch.log(torch.tanh(x) + 1.0)
        elif activation == 'relu':
            self.sigma = lambda x: torch.log(nn.functional.relu(x + 1.0))
        elif activation == 'atan':
            self.sigma = lambda x: torch.log(torch.atan(x + 1.0))
        else:
            raise ValueError(f"Invalid activation function for PSL: {activation}, must be one of ['tanh', 'relu', 'atan']!")

    def forward(self, logits, labels):
        """

        :param logits: [batch_num, item_num]
        :param labels: [batch_num]
        :return: loss
        """
        # [batch_num]
        pos_score = torch.gather(logits, dim=1, index=labels.unsqueeze(1)).squeeze(1)

        batch_size, item_num = logits.size()
        mask = torch.arange(item_num).unsqueeze(0).to(labels.device) != labels.unsqueeze(1)
        negative_scores = logits[mask].view(batch_size, -1)

        # [batch_num, item_num]
        d = negative_scores - pos_score.unsqueeze(1)

        loss = torch.logsumexp(self.sigma(d) / self.tau, dim=1).mean()

        return loss