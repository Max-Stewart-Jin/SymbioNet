"""The Mamba block of the package mamba_ssm in plain PyTorch.

Some published networks begin with `from mamba_ssm import Mamba`. That package (github.com/state-spaces/mamba)
compiles a CUDA kernel for the selective scan and is built for Linux; on a Windows machine it can usually not be
installed. The class below has the name, the arguments, the parameters (names, shapes, initialisation) and the
function of mamba_ssm.modules.mamba_simple.Mamba, and computes the same recurrence step by step:

    x, z      = in_proj(input)                                  two halves of width d_inner = expand * d_model
    x         = SiLU(causal depthwise conv(x))
    dt, B, C  = x_proj(x)                                       input-dependent step, input and output vectors
    delta     = softplus(dt_proj(dt))
    h_t       = exp(delta_t A) h_{t-1} + delta_t x_t B_t,       h_{-1} = 0,   A = -exp(A_log)
    y_t       = C_t . h_t + D x_t
    output    = out_proj(y * SiLU(z))

A state_dict of one class loads into the other. With gradients, the scan is the one of symbionet/model.py (its
backward pass recomputes the states, so the memory does not grow with their history); without gradients it keeps
only the current state. tests/test_mamba_compat.py compares the block with the reference function of mamba_ssm
(selective_scan_ref) on the same parameters.

symbionet/external.py puts this module in the place of mamba_ssm only if the real package cannot be imported, and
writes into the record of a run which of the two computed it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import SelectiveScan

__all__ = ['Mamba', 'scan']


def scan(u, dt, A, Bm, Cm, D):
    """y_t = C_t . h_t + D u_t with h_t = exp(dt_t A) h_{t-1} + dt_t u_t B_t and h_{-1} = 0.
    u, dt: (B, L, c), dt > 0; A: (c, S); Bm, Cm: (B, L, S); D: (c,)."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (u, dt, A, Bm, Cm)):
        return SelectiveScan.apply(u, dt, A, Bm, Cm) + u * D
    h = u.new_zeros(u.shape[0], u.shape[2], A.shape[1])                   # the current state (B, c, S)
    ys = []
    for t in range(u.shape[1]):
        d = dt[:, t]
        h = torch.exp(d.unsqueeze(-1) * A) * h + (d * u[:, t]).unsqueeze(-1) * Bm[:, t].unsqueeze(1)
        ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
    return torch.stack(ys, 1) + u * D


class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank='auto', dt_min=0.001, dt_max=0.1,
                 dt_init='random', dt_scale=1.0, dt_init_floor=1e-4, conv_bias=True, bias=False, use_fast_path=True,
                 layer_idx=None, device=None, dtype=None):
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model, self.d_state, self.d_conv, self.expand = d_model, d_state, d_conv, expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank
        self.use_fast_path, self.layer_idx = use_fast_path, layer_idx      # kept for the signature; not used here

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **kw)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1,
                                bias=conv_bias, **kw)
        self.activation = 'silu'
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **kw)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **kw)

        std = self.dt_rank ** -0.5 * dt_scale                # the variance of dt at initialisation
        if dt_init == 'constant':
            nn.init.constant_(self.dt_proj.weight, std)
        elif dt_init == 'random':
            nn.init.uniform_(self.dt_proj.weight, -std, std)
        else:
            raise NotImplementedError(dt_init)
        dt = torch.exp(torch.rand(self.d_inner, **kw) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        with torch.no_grad():                                # softplus(bias) = dt, between dt_min and dt_max
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **kw)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, d_model); returns the same shape."""
        if inference_params is not None:
            raise NotImplementedError('step-by-step inference is not part of this implementation')
        L = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, -1)
        x = self.act(self.conv1d(x.transpose(1, 2))[..., :L]).transpose(1, 2)
        dt, Bm, Cm = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], -1)
        y = scan(x.float(), F.softplus(self.dt_proj(dt)).float(), -torch.exp(self.A_log.float()), Bm.float(),
                 Cm.float(), self.D.float())
        return self.out_proj((y * F.silu(z)).to(hidden_states.dtype))
