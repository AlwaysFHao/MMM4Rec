# -*- coding: utf-8 -*-            
# @Author : Anonymous
# @Time : 2025/1/8

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum

def dual_ssd_discrete(X, dt, A, B, C, chunk_size=None, D=None, z=None, dt_bias=None, dt_softplus=True, dt_limit_min=0, dt_limit_max=1):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (n_heads)
        dt: (batch, length, n_heads)
        dt_bias: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    batch, length, n_heads, d_head = X.shape
    if dt_softplus is True:
        if dt_bias is not None:
            dt = F.softplus(dt + dt_bias)
        else:
            dt = F.softplus(dt)
    X = X * dt.unsqueeze(-1)
    A = A * dt
    A = rearrange(A, "b l h -> b h l")
    # [b, h, l] -> [b, h, l, l]
    L = torch.exp(segsum(A))
    # Y = torch.einsum("blhn,blhn,bhll,blhp->blhp", C, B, L, X)
    scores = torch.einsum("blhn,bshn->bhls", C, B)
    scores = torch.einsum("bhls,bhls->bhls", scores, L)
    Y = torch.einsum("bhls,bshp->blhp", scores, X)
    if D is not None:
        if D.shape[-1] == n_heads:
            Y += X * D.unsqueeze(-1)
        elif D.shape[-1] == d_head:
            Y += X * D
        else:
            raise ValueError(f'The dimension of D does not meet the requirements! {D.shape} must be [{n_heads}] or [{n_heads}, {d_head}]')
    return Y


if __name__ == '__main__':
    torch.manual_seed(42)

    ## Dimensions
    # Denoted (B, T, Q, D, P) in the paper
    batch, seqlen, dim, headdim = 1, 50, 64, 16
    nheads = dim // headdim  # (H) in the paper
    ngroups = 1 # (G) in the paper
    dstate = 64  # (N) in the paper
    dtype = torch.float32
    device = "cuda"
    D_has_hdim = True

    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    # dt = F.softplus(torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device) - 4).requires_grad_()
    dt = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device).requires_grad_()
    dt_bias = torch.randn(batch, seqlen, nheads, dtype=torch.float32, device=device).requires_grad_()
    A = (-torch.exp(torch.rand(nheads, dtype=torch.float32, device=device))).requires_grad_()
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device)
    D = torch.ones(dim if D_has_hdim else nheads, device=device)

    # Comparing fused version and minimal version
    y_min = dual_ssd_discrete(x, dt, A, B, C, dt_bias=dt_bias,
                              D=rearrange(D, '(h p) -> h p', p=headdim) if D_has_hdim else D)
    print(y_min.shape)
