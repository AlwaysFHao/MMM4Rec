# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/7/25
import warnings

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from .complex import ComplexLinear, ComplexAct
    from .ssd_kernel import dual_ssd_discrete as mamba_chunk_scan_combined
except:
    from complex import ComplexLinear, ComplexAct
    from ssd_kernel import dual_ssd_discrete as mamba_chunk_scan_combined

class TiSSD(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 d_state: int = 128,
                 d_conv=4,
                 conv_init=None,
                 expand: int = 2,
                 head_dim: int = 64,
                 ngroups: int=1,
                 d_ssm=None,
                 A_init_range=(1, 16),
                 D_has_hdim: bool = False,
                 rms_norm: bool = True,
                 norm_before_gate: bool = False,
                 dt_min: int = 0.001,
                 dt_max: int = 0.1,
                 dt_init_floor: float = 1e-4,
                 dt_limit=(0.0, float("inf")),
                 bias: bool = True,
                 conv_bias: bool = True,
                 chunk_size: int = 256,
                 time_drop_out: float = 0.0,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12,
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.ngroups = ngroups

        self.d_inner = self.expand * self.d_model

        self.head_dim = head_dim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm

        assert self.d_ssm % self.head_dim == 0, f'd_ssm {self.d_ssm} is not divisible by head_dim {self.head_dim}!'
        self.n_heads = self.d_ssm // self.head_dim

        if self.n_heads % 8 != 0:
            self.use_equivalent_conv1d = True
            warnings.warn(f'n_heads {self.n_heads} not divisible by 8, actually use \'nn.Conv1d\'!')
        else:
            self.use_equivalent_conv1d = False

        self.D_has_hdim = D_has_hdim
        self.rms_norm = rms_norm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = 'silu'
        self.chunk_size = chunk_size
        self.is_time = is_time

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        # Computed convolution dimension
        conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state
        self.conv1d_1 = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1
        )
        # self.conv1d_2 = nn.Conv1d(
        #     in_channels=conv_dim,
        #     out_channels=conv_dim,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=conv_dim,
        #     padding=d_conv - 1
        # )

        if is_time:
            self.conv1d4time_1 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            self.conv1d4time_2 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            # [batch_size, n_heads, seq_len] -> [batch, n_heads, seq_len]
            self.mlp4time_scale = nn.Sequential(
                nn.Dropout(p=time_drop_out),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                nn.Tanhshrink(),
            )

            self.time_gate_residual = nn.Sequential(
                nn.Dropout(p=time_drop_out),
                nn.Linear(in_features=seq_len, out_features=(seq_len if p2p_residual else 1)),
                nn.Sigmoid()
            )

            self.time_layer_norm = nn.LayerNorm(seq_len, eps=norm_eps)

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d_1.weight, -self.conv_init, self.conv_init)

        if causal_conv1d_fn is None or self.use_equivalent_conv1d:
            if self.activation == 'silu':
                self.act = nn.SiLU(inplace=True)
            else:
                self.act = nn.Hardswish(inplace=True)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert 0 < A_init_range[0] <= A_init_range[1]
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads))
        self.D._no_weight_decay = True

        if self.rms_norm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=norm_eps, norm_before_gate=self.norm_before_gate)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, time_diff=None):
        """

        :param u: [batch_size, seq_len, d_model]
        :param time_diff: [batch_size, seq_len]
        :return:
        """
        assert self.is_time is not True or time_diff is not None, 'When is_time is True, time_diff cannot be None!'
        z_x_B_C_dt = self.in_proj(u)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (n_heads) or (d_inner, d_state)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # d_mlp = d_inner - d_ssm
        d_mlp = (z_x_B_C_dt.shape[-1] - 2 * self.d_ssm - 2 * self.ngroups * self.d_state - self.n_heads) // 2

        z0, x0, z, xBC, dt = torch.split(
            z_x_B_C_dt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.ngroups * self.d_state, self.n_heads],
            dim=-1
        )

        if causal_conv1d_fn is None or self.use_equivalent_conv1d or self.activation not in ["silu", "swish"]:
            xBC = self.act(
                self.conv1d_1(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
            )
            # xBC = self.act(
            #     self.conv1d_2(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
            # )

            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                # [batch_size, num_heads, 1] * [batch_size, num_heads, seq_len] -> [batch_size, num_heads, seq_len]
                time_dt = time_diff * time_scale

                time_dt = self.act(
                    self.conv1d4time_1(time_dt)[:, :, :-(self.d_conv - 1)]
                )
                time_dt = self.act(
                    self.conv1d4time_2(time_dt).transpose(1, 2)[:, :-(self.d_conv - 1), :]
                )
                final_dt = dt * time_dt
            else:
                final_dt = dt

        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2).contiguous(),
                rearrange(self.conv1d_1.weight, 'd 1 w -> d w'),
                bias=self.conv1d_1.bias,
                activation=self.activation
            ).transpose(1, 2)
            # xBC = causal_conv1d_fn(
            #     xBC.transpose(1, 2).contiguous(),
            #     rearrange(self.conv1d_2.weight, 'd 1 w -> d w'),
            #     bias=self.conv1d_2.bias,
            #     activation=self.activation
            # ).transpose(1, 2)

            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                time_dt = time_scale * time_diff

                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_1.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_1.bias,
                    activation=self.activation
                )
                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_2.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_2.bias,
                    activation=self.activation
                ).transpose(1, 2)
                final_dt = dt * time_dt
            else:
                final_dt = dt

        x, B, C = torch.split(xBC, [self.d_ssm, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, 'b l (h p) -> b l h p', p=self.head_dim),
            final_dt,
            A,
            rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups),
            rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, '(h p) -> h p', p=self.head_dim) if self.D_has_hdim else self.D,
            z=rearrange(z, 'b l (h p) -> b l h p', p=self.head_dim) if not self.rms_norm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs
        )

        y = rearrange(y, 'b l h p -> b l (h p)')

        if self.rms_norm:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        if self.is_time:
            time_gate = self.time_gate_residual(time_diff)
            time_dt = self.time_layer_norm(time_gate * time_diff + (1 - time_gate) * time_dt.transpose(-1, -2))
            return out, time_dt
        else:
            return out, None

# class CoTiSSD(nn.Module):
#     def __init__(self,
#                  d_model: int,
#                  seq_len: int,
#                  d_state: int = 128,
#                  d_conv=4,
#                  conv_init=None,
#                  expand: int = 2,
#                  head_dim: int = 64,
#                  d_ssm=None,
#                  A_init_range=(1, 16),
#                  D_has_hdim: bool = False,
#                  rms_norm: bool = True,
#                  norm_before_gate: bool = False,
#                  dt_min: int = 0.001,
#                  dt_max: int = 0.1,
#                  dt_init_floor: float = 1e-4,
#                  dt_limit=(0.0, float("inf")),
#                  bias: bool = True,
#                  conv_bias: bool = True,
#                  chunk_size: int = 256,
#                  time_drop_out: float = 0.0,
#                  is_time: bool = True,
#                  p2p_residual: bool = False,
#                  norm_eps: float = 1e-12,
#                  ):
#         super(CoTiSSD, self).__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.conv_init = conv_init
#         self.expand = expand
#
#         self.d_inner = self.expand * self.d_model
#
#         self.head_dim = head_dim
#         self.d_ssm = self.d_inner if d_ssm is None else d_ssm
#
#         assert self.d_ssm % self.head_dim == 0, f'd_ssm {self.d_ssm} is not divisible by head_dim {self.head_dim}!'
#         self.n_heads = self.d_ssm // self.head_dim
#
#         if self.n_heads % 8 != 0:
#             self.use_equivalent_conv1d = True
#             warnings.warn(f'n_heads {self.n_heads} not divisible by 8, actually use \'nn.Conv1d\'!')
#         else:
#             self.use_equivalent_conv1d = False
#
#         self.D_has_hdim = D_has_hdim
#         self.rms_norm = rms_norm
#         self.norm_before_gate = norm_before_gate
#         self.dt_limit = dt_limit
#         self.activation = 'swish'
#         self.chunk_size = chunk_size
#         self.is_time = is_time
#
#         # Order: [x, B, dt]
#         u_d_in_proj = self.d_inner + self.d_state + self.n_heads
#         self.u_in_proj = nn.Linear(self.d_model, u_d_in_proj, bias=bias)
#
#         # Order: [z, C]
#         q_d_in_proj = self.d_inner + self.d_state
#         self.q_in_proj = nn.Linear(self.d_model, q_d_in_proj, bias=bias)
#
#         # Computed convolution dimension
#         u_conv_dim = self.d_ssm + self.d_state
#         self.u_conv1d = nn.Conv1d(
#             in_channels=u_conv_dim,
#             out_channels=u_conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=u_conv_dim,
#             padding=d_conv - 1
#         )
#
#         q_conv_dim = self.d_state
#         self.q_conv1d = nn.Conv1d(
#             in_channels=q_conv_dim,
#             out_channels=q_conv_dim,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=q_conv_dim,
#             padding=d_conv - 1
#         )
#
#         if is_time:
#             self.conv1d4time_1 = nn.Conv1d(
#                 in_channels=self.n_heads,
#                 out_channels=self.n_heads,
#                 kernel_size=d_conv,
#                 bias=conv_bias,
#                 groups=self.n_heads,
#                 padding=d_conv - 1
#             )
#
#             self.conv1d4time_2 = nn.Conv1d(
#                 in_channels=self.n_heads,
#                 out_channels=self.n_heads,
#                 kernel_size=d_conv,
#                 bias=conv_bias,
#                 groups=self.n_heads,
#                 padding=d_conv - 1
#             )
#
#             # [batch_size, n_heads, seq_len] -> [batch, n_heads, seq_len]
#             self.mlp4time_scale = nn.Sequential(
#                 nn.Dropout(p=time_drop_out),
#                 nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
#                 nn.SiLU(inplace=True),
#                 nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
#                 nn.Tanhshrink(),
#             )
#
#             # self.time_gate_residual = nn.Sequential(
#             #     nn.Dropout(p=time_drop_out),
#             #     nn.Linear(in_features=seq_len, out_features=(seq_len if p2p_residual else 1)),
#             #     nn.Sigmoid()
#             # )
#
#             # self.time_layer_norm = nn.LayerNorm(seq_len, eps=norm_eps)
#
#         if self.conv_init is not None:
#             nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
#
#         if causal_conv1d_fn is None or self.use_equivalent_conv1d:
#             self.act = nn.Hardswish(inplace=True)
#
#         # Initialize log dt bias
#         dt = torch.exp(
#             torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
#         )
#         dt = torch.clamp(dt, min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         self.dt_bias = nn.Parameter(inv_dt)
#         # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
#         # name.endswith("bias") in param_grouping.py
#         self.dt_bias._no_weight_decay = True
#
#         assert 0 < A_init_range[0] <= A_init_range[1]
#         A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(*A_init_range)
#         A_log = torch.log(A)
#         self.A_log = nn.Parameter(A_log)
#         self.A_log._no_weight_decay = True
#
#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads))
#         self.D._no_weight_decay = True
#
#         if self.rms_norm:
#             assert RMSNormGated is not None
#             self.norm = RMSNormGated(self.d_ssm, eps=norm_eps, norm_before_gate=self.norm_before_gate)
#
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
#
#     def forward(self, q, u, time_diff=None):
#         """
#         :param q: [batch_size, seq_len, d_model]
#         :param u: [batch_size, seq_len, d_model]
#         :param time_diff: [batch_size, seq_len]
#         :return:
#         """
#         assert self.is_time is not True or time_diff is not None, 'When is_time is True, time_diff cannot be None!'
#         x_B_dt = self.u_in_proj(u)
#         z_C = self.q_in_proj(q)
#         # If the model is loaded in fp16, without the .float() here, A might be -inf
#         A = -torch.exp(self.A_log.float())  # (n_heads) or (d_inner, d_state)
#
#         dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
#
#         # d_mlp = d_inner - d_ssm
#         d_mlp = (x_B_dt.shape[-1] - self.d_ssm - self.d_state - self.n_heads)
#
#         x0, xB, dt = torch.split(
#             x_B_dt,
#             [d_mlp, self.d_ssm + self.d_state, self.n_heads],
#             dim=-1
#         )
#
#         z0, z, C = torch.split(
#             z_C,
#             [d_mlp, self.d_ssm, self.d_state],
#             dim=-1
#         )
#
#         if causal_conv1d_fn is None or self.use_equivalent_conv1d or self.activation not in ["silu", "swish"]:
#             xB = self.act(
#                 self.u_conv1d(xB.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
#             )
#             C = self.act(
#                 self.q_conv1d(C.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
#             )
#
#             if self.is_time:
#                 time_scale = self.mlp4time_scale(time_diff)
#                 # [batch_size, num_heads, 1] * [batch_size, num_heads, seq_len] -> [batch_size, num_heads, seq_len]
#                 time_dt = time_diff * time_scale
#
#                 time_dt = self.act(
#                     self.conv1d4time_1(time_dt)[:, :, :-(self.d_conv - 1)]
#                 )
#                 time_dt = self.act(
#                     self.conv1d4time_2(time_dt).transpose(1, 2)[:, :-(self.d_conv - 1), :]
#                 )
#                 final_dt = dt * time_dt
#             else:
#                 final_dt = dt
#
#         else:
#             xB = causal_conv1d_fn(
#                 xB.transpose(1, 2).contiguous(),
#                 rearrange(self.u_conv1d.weight, 'd 1 w -> d w'),
#                 bias=self.u_conv1d.bias,
#                 activation=self.activation
#             ).transpose(1, 2)
#
#             C = causal_conv1d_fn(
#                 C.transpose(1, 2).contiguous(),
#                 rearrange(self.q_conv1d.weight, 'd 1 w -> d w'),
#                 bias=self.q_conv1d.bias,
#                 activation=self.activation
#             ).transpose(1, 2)
#
#
#             if self.is_time:
#                 time_scale = self.mlp4time_scale(time_diff)
#                 time_dt = time_scale * time_diff
#
#                 time_dt = causal_conv1d_fn(
#                     time_dt,
#                     rearrange(self.conv1d4time_1.weight, 'd 1 w -> d w'),
#                     bias=self.conv1d4time_1.bias,
#                     activation=self.activation
#                 )
#                 time_dt = causal_conv1d_fn(
#                     time_dt,
#                     rearrange(self.conv1d4time_2.weight, 'd 1 w -> d w'),
#                     bias=self.conv1d4time_2.bias,
#                     activation=self.activation
#                 ).transpose(1, 2)
#                 final_dt = dt * time_dt
#             else:
#                 final_dt = dt
#
#         x, B = torch.split(xB, [self.d_ssm, self.d_state], dim=-1)
#         y = mamba_chunk_scan_combined(
#             rearrange(x, 'b l (h p) -> b l h p', p=self.head_dim),
#             final_dt,
#             A,
#             rearrange(B, 'b l (g n) -> b l g n', g=1),
#             rearrange(C, 'b l (g n) -> b l g n', g=1),
#             chunk_size=self.chunk_size,
#             D=rearrange(self.D, '(h p) -> h p', p=self.head_dim) if self.D_has_hdim else self.D,
#             z=rearrange(z, 'b l (h p) -> b l h p', p=self.head_dim) if not self.rms_norm else None,
#             dt_bias=self.dt_bias,
#             dt_softplus=True,
#             **dt_limit_kwargs
#         )
#
#         y = rearrange(y, 'b l h p -> b l (h p)')
#
#         if self.rms_norm:
#             y = self.norm(y, z)
#
#         if d_mlp > 0:
#             y = torch.cat([F.silu(z0) * x0, y], dim=-1)
#
#         out = self.out_proj(y)
#
#         # if self.is_time:
#         #     time_gate = self.time_gate_residual(time_diff)
#         #     time_dt = self.time_layer_norm(time_gate * time_diff + (1 - time_gate) * time_dt.transpose(-1, -2))
#         #     return out, time_dt
#         # else:
#         #     return out, None
#
#         return out

class CoTiSSD(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 d_state: int = 128,
                 d_conv=4,
                 conv_init=None,
                 expand: int = 2,
                 head_dim: int = 64,
                 ngroups: int = 1,
                 d_ssm=None,
                 A_init_range=(1, 16),
                 D_has_hdim: bool = False,
                 rms_norm: bool = True,
                 norm_before_gate: bool = False,
                 dt_min: int = 0.001,
                 dt_max: int = 0.1,
                 dt_init_floor: float = 1e-4,
                 dt_limit=(0.0, float("inf")),
                 bias: bool = True,
                 conv_bias: bool = True,
                 chunk_size: int = 256,
                 time_drop_out: float = 0.0,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12,
                 ):
        super(CoTiSSD, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.ngroups = ngroups

        self.d_inner = self.expand * self.d_model

        self.head_dim = head_dim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm

        assert self.d_ssm % self.head_dim == 0, f'd_ssm {self.d_ssm} is not divisible by head_dim {self.head_dim}!'
        self.n_heads = self.d_ssm // self.head_dim

        if self.n_heads % 8 != 0:
            self.use_equivalent_conv1d = True
            warnings.warn(f'n_heads {self.n_heads} not divisible by 8, actually use \'nn.Conv1d\'!')
        else:
            self.use_equivalent_conv1d = False

        self.D_has_hdim = D_has_hdim
        self.rms_norm = rms_norm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = 'swish'
        self.chunk_size = chunk_size
        self.is_time = is_time

        # Order: [z, x, B, dt]
        u_d_in_proj = self.d_inner + self.ngroups * self.d_state + self.n_heads
        self.u_in_proj = nn.Linear(self.d_model, u_d_in_proj, bias=bias)
        self.z_in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)

        # C
        q_d_in_proj = self.ngroups * self.d_state
        self.q_in_proj = nn.Linear(self.d_model, q_d_in_proj, bias=bias)

        # Computed convolution dimension
        u_conv_dim = self.d_ssm + self.ngroups * self.d_state
        self.u_conv1d = nn.Conv1d(
            in_channels=u_conv_dim,
            out_channels=u_conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=u_conv_dim,
            padding=d_conv - 1
        )

        q_conv_dim = self.ngroups * self.d_state
        self.q_conv1d = nn.Conv1d(
            in_channels=q_conv_dim,
            out_channels=q_conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=q_conv_dim,
            padding=d_conv - 1
        )

        if is_time:
            self.conv1d4time_1 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            self.conv1d4time_2 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            # [batch_size, n_heads, seq_len] -> [batch, n_heads, seq_len]
            self.mlp4time_scale = nn.Sequential(
                nn.Dropout(p=time_drop_out),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                nn.Tanhshrink(),
            )

            # self.time_gate_residual = nn.Sequential(
            #     nn.Dropout(p=time_drop_out),
            #     nn.Linear(in_features=seq_len, out_features=(seq_len if p2p_residual else 1)),
            #     nn.Sigmoid()
            # )

            # self.time_layer_norm = nn.LayerNorm(seq_len, eps=norm_eps)

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if causal_conv1d_fn is None or self.use_equivalent_conv1d:
            self.act = nn.Hardswish(inplace=True)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert 0 < A_init_range[0] <= A_init_range[1]
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads))
        self.D._no_weight_decay = True

        if self.rms_norm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=norm_eps, norm_before_gate=self.norm_before_gate)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, q, u, time_diff=None):
        """
        :param q: [batch_size, seq_len, d_model]
        :param u: [batch_size, seq_len, d_model]
        :param time_diff: [batch_size, seq_len]
        :return:
        """
        assert self.is_time is not True or time_diff is not None, 'When is_time is True, time_diff cannot be None!'

        x_B_dt = self.u_in_proj(u)
        z = self.z_in_proj(q + u)
        C = self.q_in_proj(q)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (n_heads) or (d_inner, d_state)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # d_mlp = d_inner - d_ssm
        d_mlp = (x_B_dt.shape[-1] - self.d_ssm - self.ngroups * self.d_state - self.n_heads)

        x0, xB, dt = torch.split(
            x_B_dt,
            [d_mlp, self.d_ssm + self.ngroups * self.d_state, self.n_heads],
            dim=-1
        )

        z0, z = torch.split(
            z,
            [d_mlp, self.d_ssm],
            dim=-1
        )

        if causal_conv1d_fn is None or self.use_equivalent_conv1d or self.activation not in ["silu", "swish"]:
            xB = self.act(
                self.u_conv1d(xB.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
            )
            C = self.act(
                self.q_conv1d(C.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
            )

            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                # [batch_size, num_heads, 1] * [batch_size, num_heads, seq_len] -> [batch_size, num_heads, seq_len]
                time_dt = time_diff * time_scale

                time_dt = self.act(
                    self.conv1d4time_1(time_dt)[:, :, :-(self.d_conv - 1)]
                )
                time_dt = self.act(
                    self.conv1d4time_2(time_dt).transpose(1, 2)[:, :-(self.d_conv - 1), :]
                )
                final_dt = dt * time_dt
            else:
                final_dt = dt

        else:
            xB = causal_conv1d_fn(
                xB.transpose(1, 2).contiguous(),
                rearrange(self.u_conv1d.weight, 'd 1 w -> d w'),
                bias=self.u_conv1d.bias,
                activation=self.activation
            ).transpose(1, 2)

            C = causal_conv1d_fn(
                C.transpose(1, 2).contiguous(),
                rearrange(self.q_conv1d.weight, 'd 1 w -> d w'),
                bias=self.q_conv1d.bias,
                activation=self.activation
            ).transpose(1, 2)


            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                time_dt = time_scale * time_diff

                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_1.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_1.bias,
                    activation=self.activation
                )
                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_2.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_2.bias,
                    activation=self.activation
                ).transpose(1, 2)
                final_dt = dt * time_dt
            else:
                final_dt = dt

        x, B = torch.split(xB, [self.d_ssm, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, 'b l (h p) -> b l h p', p=self.head_dim),
            final_dt,
            A,
            rearrange(B, 'b l (g n) -> b l g n', g=self.ngroups),
            rearrange(C, 'b l (g n) -> b l g n', g=self.ngroups),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, '(h p) -> h p', p=self.head_dim) if self.D_has_hdim else self.D,
            z=rearrange(z, 'b l (h p) -> b l h p', p=self.head_dim) if not self.rms_norm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs
        )

        y = rearrange(y, 'b l h p -> b l (h p)')

        if self.rms_norm:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        # if self.is_time:
        #     time_gate = self.time_gate_residual(time_diff)
        #     time_dt = self.time_layer_norm(time_gate * time_diff + (1 - time_gate) * time_dt.transpose(-1, -2))
        #     return out, time_dt
        # else:
        #     return out, None

        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2, norm_eps=1e-12):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, inner_size)
        self.fc2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Hardswish()
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

    def forward(self, x):
        hidden = self.act(self.fc1(x))
        hidden = self.dropout(hidden)

        hidden = self.fc2(hidden)
        hidden = self.layer_norm(self.dropout(hidden) + x)
        return hidden

class TiSSDLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 inner_size: int,
                 seq_len: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 num_layers: int,
                 head_dim: int,
                 ngroups: int,
                 chunk_size: int,
                 dropout: float,
                 time_drop_out: float,
                 is_ffn: bool = True,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12):
        """
        A single-layer TiSSDLayer, containing a TiSSDBlock and an FFN(if is_ffn is True)

        :param d_model: vector embedding dimension
        :param d_model: ffn inner dimension
        :param d_state: the B, C matrix dimension in SSD
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param num_layers: the number of SSDLayer layers,
                used to determined whether the SSDLayer needs residuals connections
        :param head_dim: Header dimension of an SSD
        :param chunk_size: Chunk size of an SSD
        :param dropout: dropout_radio
        :param time_drop_out: time_dropout_radio
        :param is_ffn: whether the FFN is included
        :param is_time: whether the Time-aware is included
        :param p2p_residual: whether you use point-to-point residuals
        :param norm_eps: normalization epsilon
        """
        super(TiSSDLayer, self).__init__()
        self.num_layers = num_layers
        self.ssd = TiSSD(
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            time_drop_out=time_drop_out,
            is_time=is_time,
            p2p_residual=p2p_residual,
            norm_eps=norm_eps
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = FeedForward(
                d_model=d_model,
                inner_size=inner_size,
                dropout=dropout
            )

    def forward(self, x, time_diff=None):
        """
        x -> ssd(x)
        -> ffn(x) if is_ffn is True
        :param x: shape: [batch_size, seq_len, d_model]
        :param time_diff: shape: [batch_size, seq_len]
        :return: shape: [batch_size, seq_len, d_model]
        """
        # hidden = self.layer_norm(x)
        hidden, time_diff = self.ssd(x, time_diff)

        hidden = self.layer_norm(self.dropout(hidden) + x)

        # Determine whether SSDBlock needs residual by num_layers
        # if self.num_layers == 1:
        #     hidden = self.layer_norm(self.dropout(hidden))
        # else:
        #     hidden = self.layer_norm(self.dropout(hidden) + x)

        if self.is_ffn:
            return self.ffn(hidden), time_diff
        else:
            return hidden, time_diff


class CoTiSSDLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 inner_size: int,
                 seq_len: int,
                 d_state: int,
                 d_conv: int,
                 expand: int,
                 num_layers: int,
                 head_dim: int,
                 ngroups: int,
                 chunk_size: int,
                 dropout: float,
                 time_drop_out: float,
                 is_ffn: bool = True,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12,
                 ssd_block=None):
        """
        A single-layer TiSSDLayer, containing a TiSSDBlock and an FFN(if is_ffn is True)

        :param d_model: vector embedding dimension
        :param d_model: ffn inner dimension
        :param d_state: the B, C matrix dimension in SSD
        :param d_conv: causal-conv1d kernel size
        :param expand: coefficient of expanding
        :param num_layers: the number of SSDLayer layers,
                used to determined whether the SSDLayer needs residuals connections
        :param head_dim: Header dimension of an SSD
        :param chunk_size: Chunk size of an SSD
        :param dropout: dropout_radio
        :param time_drop_out: time_dropout_radio
        :param is_ffn: whether the FFN is included
        :param is_time: whether the Time-aware is included
        :param p2p_residual: whether you use point-to-point residuals
        :param norm_eps: normalization epsilon
        """
        super(CoTiSSDLayer, self).__init__()
        self.num_layers = num_layers
        self.ssd = TiSSD(
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            time_drop_out=time_drop_out,
            is_time=is_time,
            p2p_residual=p2p_residual,
            norm_eps=norm_eps
        ) if ssd_block is None else ssd_block
        self.co_ssd = CoTiSSD(
            d_model=d_model,
            seq_len=seq_len,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            head_dim=head_dim,
            ngroups=ngroups,
            chunk_size=chunk_size,
            bias=True,
            rms_norm=True,
            time_drop_out=time_drop_out,
            is_time=is_time,
            p2p_residual=p2p_residual,
            norm_eps=norm_eps
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.co_layer_norm = nn.LayerNorm(d_model, eps=norm_eps)

        self.is_time = is_time
        if is_time:
            self.vision_complex_weight = nn.Sequential(
                ComplexLinear(in_features=seq_len // 2 + 1, out_features=seq_len // 2 + 1),
                # ComplexAct(act=nn.functional.relu),
                # ComplexLinear(in_features=seq_len // 2 + 1, out_features=seq_len // 2 + 1),
                # ComplexAct(act=nn.functional.sigmoid)
            )

            self.text_complex_weight = nn.Sequential(
                ComplexLinear(in_features=seq_len // 2 + 1, out_features=seq_len // 2 + 1),
                # ComplexAct(act=nn.functional.relu),
                # ComplexLinear(in_features=seq_len // 2 + 1, out_features=seq_len // 2 + 1),
                # ComplexAct(act=nn.functional.sigmoid)
            )

            self.co_complex_weight = nn.Sequential(
                ComplexLinear(in_features=seq_len // 2 + 1, out_features=seq_len // 2 + 1)
            )

        self.is_ffn = is_ffn
        if is_ffn:
            self.ffn = FeedForward(
                d_model=d_model,
                inner_size=inner_size,
                dropout=dropout
            )

    def forward(self, encoder_input, decoder_input, encoder_time_diff=None, decoder_time_diff=None):
        """
        :param encoder_input: shape: [batch_size, seq_len, d_model]
        :param decoder_input: shape: [batch_size, seq_len, d_model]
        :param decoder_time_diff: shape: [batch_size, seq_len]
        :param encoder_time_diff: shape: [batch_size, seq_len]
        :return: shape: [batch_size, seq_len, d_model]
        """
        # hidden = self.layer_norm(x)
        hidden, decoder_time_diff = self.ssd(decoder_input, decoder_time_diff)
        # Determine whether SSDBlock needs residual by num_layers
        hidden = self.layer_norm(self.dropout(hidden) + decoder_input)
        # if self.num_layers == 1:
        #     hidden = self.layer_norm(self.dropout(hidden))
        # else:
        #     hidden = self.layer_norm(self.dropout(hidden) + encoder_input)

        # hidden, text_time_diff = self.co_ssd(q, hidden, text_time_diff)
        co_time_diff = None
        if self.is_time:
            seq_len = decoder_time_diff.shape[-1]
            temp_decoder_time_diff = torch.fft.rfft(decoder_time_diff, dim=2, norm='ortho')
            text_weight = self.text_complex_weight(temp_decoder_time_diff)
            temp_decoder_time_diff = text_weight * temp_decoder_time_diff
            # temp_text_time_diff = self.text_complex_weight(temp_text_time_diff)


            encoder_time_diff = torch.fft.rfft(encoder_time_diff, dim=2, norm='ortho')
            encoder_weight = self.vision_complex_weight(encoder_time_diff)
            encoder_time_diff = encoder_weight * encoder_time_diff
            # vision_time_diff = self.vision_complex_weight(vision_time_diff)

            co_time_diff = (temp_decoder_time_diff + encoder_time_diff)
            co_time_diff = self.co_complex_weight(co_time_diff)
            # co_time_diff = co_time_diff_weight * co_time_diff

            co_time_diff = torch.fft.irfft(co_time_diff, n=seq_len, dim=2, norm='ortho')

        output = self.co_ssd(q=hidden, u=encoder_input, time_diff=co_time_diff)

        output = self.co_layer_norm(self.dropout(output) + hidden + encoder_input)

        # if self.num_layers == 1:
        #     output = self.co_layer_norm(self.dropout(output))
        # else:
        #     output = self.co_layer_norm(self.dropout(output) + hidden + encoder_input)

        if self.is_ffn:
            return self.ffn(output), decoder_time_diff
        else:
            return output, decoder_time_diff


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q_test = torch.randn(8, 50, 64).to(device)
    x_test = torch.randn(8, 50, 64).to(device)
    n_heads = 8
    h_dim = 64 * 2 // n_heads
    time_test = torch.randn(8, n_heads, 50).to(device)
    model = CoTiSSD(d_model=64, d_state=32, d_conv=4, expand=2, head_dim=h_dim, seq_len=50).to(device)
    y_test, _ = model(q_test, x_test, time_test)
    print(y_test.shape)
