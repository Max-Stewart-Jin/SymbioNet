"""Tests of symbionet/mamba_compat.py: the block against the reference function of mamba_ssm, its two ways of
scanning against each other, its gradients, and its parameters. Run with `python tests/test_mamba_compat.py` or
with pytest.

The comparison with mamba_ssm reads the function selective_scan_ref from a copy of the official source
(mamba_ssm/ops/selective_scan_interface.py). Give the folder of the copy in the environment variable MAMBA_SSM_SRC,
or put it under external/; without a copy that one test is skipped.
"""
import ast
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from symbionet.mamba_compat import Mamba, scan  # noqa: E402

REF = 'mamba_ssm/ops/selective_scan_interface.py'


def official_reference():
    """selective_scan_ref of the official source, or None when no copy of the source (or einops) is at hand."""
    places = [os.environ.get('MAMBA_SSM_SRC'), ROOT / 'external']
    for place in [Path(p) for p in places if p]:
        for f in [place / REF] + sorted(place.glob('*/' + REF)) + sorted(place.glob('*/*/' + REF)):
            if f.exists():
                try:
                    from einops import rearrange, repeat
                except ImportError:
                    return None
                tree = ast.parse(f.read_text(encoding='utf-8'))
                node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'selective_scan_ref')
                space = dict(torch=torch, F=F, rearrange=rearrange, repeat=repeat)
                exec(compile(ast.Module([node], []), str(f), 'exec'), space)      # the function only, nothing else
                return space['selective_scan_ref']
    return None


def reference_forward(m: Mamba, hidden, selective_scan_ref):
    """The forward pass of mamba_ssm.Mamba without its fused kernel (the `else` branch of its forward), with the
    parameters of m and the official reference scan. Layout as there: (B, d, L)."""
    L = hidden.shape[1]
    xz = m.in_proj(hidden).transpose(1, 2)
    x, z = xz.chunk(2, dim=1)
    x = F.silu(m.conv1d(x)[..., :L])
    x_dbl = m.x_proj(x.transpose(1, 2))                                      # (B, L, rank + 2 S)
    dt, Bm, Cm = torch.split(x_dbl, [m.dt_rank, m.d_state, m.d_state], dim=-1)
    dt = (dt @ m.dt_proj.weight.t()).transpose(1, 2)                         # (B, d, L), bias added in the scan
    y = selective_scan_ref(x, dt, -torch.exp(m.A_log.float()), Bm.transpose(1, 2).contiguous(),
                           Cm.transpose(1, 2).contiguous(), m.D.float(), z=z, delta_bias=m.dt_proj.bias.float(),
                           delta_softplus=True)
    return m.out_proj(y.transpose(1, 2))


def test_against_the_reference_of_mamba_ssm():
    ref = official_reference()
    if ref is None:
        print('   (no copy of the mamba_ssm source: the comparison with selective_scan_ref is skipped)')
        return
    torch.manual_seed(0)
    for kw in (dict(d_model=64, d_state=32, d_conv=4, expand=2), dict(d_model=24, d_state=16),
               dict(d_model=16, d_state=8, d_conv=3, expand=1, bias=True, conv_bias=False)):
        m = Mamba(**kw).double().eval()
        with torch.no_grad():                                                # away from the initial values
            for p in m.parameters():
                p.add_(0.05 * torch.randn_like(p))
        x = torch.randn(3, 29, kw['d_model'], dtype=torch.double)
        want = reference_forward(m, x, ref).double()
        with torch.no_grad():
            lean = m(x)                                                      # the scan that keeps one state
        full = m(x)                                                          # the scan with a backward pass
        scale = want.abs().max().item()
        assert (lean - want).abs().max().item() < 1e-4 * scale, (kw, (lean - want).abs().max().item(), scale)
        assert (full - want).abs().max().item() < 1e-4 * scale, (kw, (full - want).abs().max().item(), scale)


def test_two_scans_agree_and_gradients():
    torch.manual_seed(1)
    B, L, c, S = 2, 7, 3, 4
    u = torch.randn(B, L, c, dtype=torch.double, requires_grad=True)
    dt = (0.05 + torch.rand(B, L, c, dtype=torch.double)).requires_grad_()
    A = (-torch.rand(c, S, dtype=torch.double) - 0.1).requires_grad_()
    Bm = torch.randn(B, L, S, dtype=torch.double, requires_grad=True)
    Cm = torch.randn(B, L, S, dtype=torch.double, requires_grad=True)
    D = torch.randn(c, dtype=torch.double, requires_grad=True)
    with_grad = scan(u, dt, A, Bm, Cm, D)
    with torch.no_grad():
        lean = scan(u, dt, A, Bm, Cm, D)
    assert (with_grad - lean).abs().max().item() < 1e-12
    h = torch.zeros(B, c, S, dtype=torch.double)                             # the recurrence written out
    for t in range(L):
        h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t] * u[:, t])[..., None] * Bm[:, t, None, :]
        y = (h * Cm[:, t, None, :]).sum(-1) + D * u[:, t]
        assert (y - lean[:, t]).abs().max().item() < 1e-12
    assert torch.autograd.gradcheck(scan, (u, dt, A, Bm, Cm, D), eps=1e-6, atol=1e-6)


def test_parameters_as_in_mamba_ssm():
    torch.manual_seed(2)
    m = Mamba(d_model=64, d_state=32, d_conv=4, expand=2)
    shapes = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes == {'A_log': (128, 32), 'D': (128,), 'in_proj.weight': (256, 64), 'conv1d.weight': (128, 1, 4),
                      'conv1d.bias': (128,), 'x_proj.weight': (4 + 64, 128), 'dt_proj.weight': (128, 4),
                      'dt_proj.bias': (128,), 'out_proj.weight': (64, 128)}, shapes
    assert m.dt_rank == math.ceil(64 / 16) and m.d_inner == 128
    dt = F.softplus(m.dt_proj.bias)
    assert dt.min().item() >= 1e-3 - 1e-6 and dt.max().item() <= 1e-1 + 1e-6      # the initial step sizes
    assert m.dt_proj.weight.abs().max().item() <= 4 ** -0.5 + 1e-6
    assert torch.allclose(torch.exp(m.A_log[0]), torch.arange(1.0, 33.0)) and bool((m.D == 1).all())
    x = torch.randn(5, 121, 64)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    m.eval()
    with torch.no_grad():                                    # causal: an output does not depend on later tokens
        a = m(x)
        x2 = x.clone()
        x2[:, 60:] += 1.0
        b = m(x2)
    assert (a[:, :60] - b[:, :60]).abs().max().item() < 1e-5 and (a[:, 60:] - b[:, 60:]).abs().max().item() > 1e-3


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print('test_mamba_compat: %d tests passed' % len(tests))
