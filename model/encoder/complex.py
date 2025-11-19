# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/12/22
import torch
import torch.nn as nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ComplexLinear, self).__init__()
        self.real = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.imag = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

    def forward(self, x, dtype=torch.complex64):
        """
        The input needs to be complex
        :param x: [..., N]
        :param dtype: torch.complex32 or torch.complex64
        :return: [..., N]
        """
        # real
        real_input = x.real
        # imaginary
        imag_input = x.imag

        real_output = self.real(real_input) - self.imag(imag_input)
        imag_output = self.imag(real_input) + self.real(imag_input)

        return real_output.type(dtype) + 1j * imag_output.type(dtype)


class ComplexAct(nn.Module):
    def __init__(self, act):
        super(ComplexAct, self).__init__()
        self.act = act

    def forward(self, x, dtype=torch.complex64):
        return self.act(x.real).type(dtype) + 1j * self.act(x.imag)