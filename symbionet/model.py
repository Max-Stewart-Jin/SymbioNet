"""SymbioNet: gated symmetric fusion, masked labeled copies, and spatial-spectral mutual completion."""
import math
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ('h', 'l')
MASK_MODES = ('independent', 'same', 'complementary')
LOST_BIAS = -30.0
TEST_TIME_PREFIXES = ('stem.', 'pos', 'blocks.', 'norm.', 'head.')

def gelu_gradient_ok(device='cpu') -> bool:
    """True if GELU's gradient is right when the incoming gradient is not contiguous, as in the conv stems (GELU, then
    flatten and transpose into tokens). PyTorch 2.1.0 with oneDNN on the CPU gets it wrong (values from uninitialised
    memory); its plain CPU kernels and the CUDA kernels are right."""
    g = torch.Generator().manual_seed(0)
    y = torch.randn(2, 8, 4, generator=g).to(device).requires_grad_()
    up = torch.randn(2, 4, 8, generator=g).to(device)
    (F.gelu(y).transpose(1, 2) * up).sum().backward()
    x = y.detach()
    dgelu = 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * torch.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    ref = dgelu * up.transpose(1, 2)
    return bool(torch.allclose(y.grad, ref, atol=1e-5))


if torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled and not gelu_gradient_ok('cpu'):
    torch.backends.mkldnn.enabled = False
    warnings.warn('this PyTorch build computes wrong GELU gradients with oneDNN on the CPU, so oneDNN was switched off '
                  '(CPU only: training on the GPU is not affected)', RuntimeWarning)


# ----------------------------------------------------------------------------------------------- building blocks
def conv_stem(cin, mid, d):
    """Two 3x3 conv + BN + GELU layers."""
    return nn.Sequential(nn.Conv2d(cin, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
                         nn.Conv2d(mid, d, 3, padding=1), nn.BatchNorm2d(d), nn.GELU())


def mlp(d, hidden):
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))


def attention(q, k, v, heads, bias=None, kbias=None):
    """Softmax attention written out (deterministic on GPU). q: (B, n, d); k, v: (B, m, d); bias: (heads, n, m),
    added to the scores of every patch; kbias: (B, m), added to the scores of every head and query of its patch."""
    B, n, d = q.shape
    m, dh = k.shape[1], d // heads
    q = q.reshape(B, n, heads, dh).transpose(1, 2)
    k = k.reshape(B, m, heads, dh).transpose(1, 2)
    v = v.reshape(B, m, heads, dh).transpose(1, 2)
    s = q @ k.transpose(-2, -1) / math.sqrt(dh)
    if bias is not None:
        s = s + bias
    if kbias is not None:
        s = s + kbias[:, None, None, :]
    return (torch.softmax(s, -1) @ v).transpose(1, 2).reshape(B, n, d)


class SFB(nn.Module):
    """Each modality queries both modalities, with separate projection parameters."""

    def __init__(self, d=64, heads=4, hidden=128):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by heads')
        self.heads = heads
        self.n1 = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.qkv = nn.ModuleDict({m: nn.Linear(d, 3 * d) for m in MODALITIES})
        self.o = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
        self.n2 = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.mlp = nn.ModuleDict({m: mlp(d, hidden) for m in MODALITIES})

    def attend(self, m, q, k, v, kb=None):
        K, V = torch.cat([k['h'], k['l']], 1), torch.cat([v['h'], v['l']], 1)
        return attention(q[m], K, V, self.heads, None,
                         None if kb is None else torch.cat([kb['h'], kb['l']], 1))

    def forward(self, th, tl, kb=None):
        t = {'h': th, 'l': tl}
        q, k, v = {}, {}, {}
        for m in MODALITIES:
            q[m], k[m], v[m] = self.qkv[m](self.n1[m](t[m])).chunk(3, -1)
        out = []
        for m in MODALITIES:
            x = t[m] + self.o[m](self.attend(m, q, k, v, kb))
            out.append(x + self.mlp[m](self.n2[m](x)))
        return out[0], out[1]


def uniform(shape, g, device):
    """U(0, 1) draws. With a generator they are made on the generator's device (the callers move the result), so a
    CPU generator gives the same masks whether the model runs on the CPU or the GPU."""
    if g is None:
        return torch.rand(shape, device=device)
    return torch.rand(shape, generator=g, device=g.device)


def rank(B, n, g, device):
    """Per row, every position's rank in a random order."""
    return uniform((B, n), g, device).argsort(1).argsort(1).to(device)


def make_masks(B, n, ratio, mode, split, g, device):
    """1 = masked; float masks (B, n) for the HSI and the LiDAR input."""
    if mode not in MASK_MODES:
        raise ValueError('mask_mode must be one of %s' % (MASK_MODES,))
    if mode == 'same':
        m = (rank(B, n, g, device) < int(round(ratio * n))).float()
        return m, m.clone()
    V = int(round(ratio * 2 * n))
    if mode == 'complementary' and V > n:
        raise ValueError('complementary masking needs mask_ratio <= 0.5')
    if split == 'equal':
        nh = torch.full((B,), V // 2, dtype=torch.long, device=device)
    elif split == 'dirichlet':
        nh = torch.round(uniform((B,), g, device) * V).long().to(device)
    else:
        raise ValueError('split must be "dirichlet" or "equal"')
    nh = nh.clamp(max=n)
    nl = (V - nh).clamp(min=0, max=n)
    r = rank(B, n, g, device)
    mh = (r < nh[:, None]).float()
    if mode == 'complementary':
        ml = ((r >= nh[:, None]) & (r < (nh + nl)[:, None])).float()
    else:
        ml = (rank(B, n, g, device) < nl[:, None]).float()
    return mh, ml


def block_index(w, b, device):
    """Block of every pixel of a w x w patch (row-major) for b x b blocks tiled from the top-left corner: (w * w,)."""
    r = torch.arange(w, device=device) // b
    return (r[:, None] * math.ceil(w / b) + r[None, :]).flatten()


def masked_mean(e, m):
    """Mean of e over the masked slots (e, m: (B, n))."""
    return (e * m).sum() / m.sum().clamp_min(1.0)


@contextmanager
def bn_frozen(net):
    """BatchNorm layers normalise with the batch's statistics but do not update their running ones."""
    bns = [x for x in net.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    old = [x.momentum for x in bns]
    for x in bns:
        x.momentum = 0.0
    try:
        yield
    finally:
        for x, mo in zip(bns, old):
            x.momentum = mo


def draw_spectral(h, g, max_rate=0.3, p_spec=0.5):
    """Per-example U(0,max_rate) conditional on independent Bernoulli(p_spec).

    Draw a rate and band order for every sample, then draw activation using
    the same dedicated CPU generator. Inactive rows are empty.
    """
    if not 0. <= p_spec <= 1.:
        raise ValueError('Invalid spectral activation probability')
    B, C = h.shape[:2]
    k = (torch.rand(B, generator=g) * (max_rate * C)).round().long()
    order = torch.rand(B, C, generator=g).argsort(dim=1).argsort(dim=1)
    bh = order < k[:, None]
    if p_spec == 0.:
        bh.zero_()
    elif p_spec != 1.:
        active = torch.rand(B, generator=g) < p_spec
        bh = bh & active[:, None]
    return bh.to(h.device)


def apply_spectral(h_masked, mh, bh, fill_h):
    """Fill selected HSI bands with the original per-band means.

    Apply only at spatially visible sites; preserve the original spatial
    holes exactly. Both masks use the same fill as the known pre-gate and
    the original Bands evaluation. No amplitude rescaling.
    """
    spatial = mh.reshape(h_masked.shape[0], 1, *h_masked.shape[2:]).bool()
    return torch.where(bh[:, :, None, None] & ~spatial, fill_h, h_masked)


class SymbioNet(nn.Module):
    def __init__(self, ch, cl, n_classes, window=11, d=64, heads=4, hidden=128, depth=2,
                 mask_mode='complementary', mask_ratio=0.5, mask_split='dirichlet', mask_block=1,
                 beta=1.0, mask_labeled=True, pre_gate='known', pre_gate_p=0.5,
                 p_spec=None, spec_max=None, *, spectral_probability=None, spectral_max=None, **legacy):
        super().__init__()
        # Both parameter spellings describe the same spectral mask. Reject
        # conflicting values rather than silently overriding a saved setting.
        if p_spec is not None and spectral_probability is not None and p_spec != spectral_probability:
            raise ValueError('Conflicting spectral probabilities')
        if spec_max is not None and spectral_max is not None and spec_max != spectral_max:
            raise ValueError('Conflicting spectral maximum rates')
        p_spec = spectral_probability if spectral_probability is not None else (0.5 if p_spec is None else p_spec)
        spec_max = spectral_max if spectral_max is not None else (0.3 if spec_max is None else spec_max)
        # Accept the saved configuration of the same architecture. Unsupported
        # alternative architectures are rejected, never silently reinterpreted.
        compatibility = {'decoder': 'identity', 'dec_depth': 2, 'd_state': 8, 'expand': 1, 'rec_weighting': 'none', 'rec_source': 'self', 'feat': True, 'cons': False, 'cons_tau': 0.0, 'lam': 0.0, 'gamma': 1.0, 'stem': 'conv', 'rel_bias': False, 'readout': 'mean', 'fusion': 'joint', 'fill': 'mean', 'feat_head': 'linear', 'gate': False, 'band_mask': 0.0, 'bn_masked': 'batch'}
        for k, v in legacy.items():
            if k not in compatibility or compatibility[k] != v:
                raise ValueError(f'Unsupported model option: {k}={v!r}')
        if mask_mode not in MASK_MODES:
            raise ValueError('invalid mask_mode')
        if not isinstance(mask_block, int) or not 1 <= mask_block <= window:
            raise ValueError('invalid mask_block')
        if pre_gate not in (False, 'known') or not 0 < pre_gate_p <= 1:
            raise ValueError('invalid pre_gate setting')
        if not 0 <= p_spec <= 1 or not 0 <= spec_max <= 1:
            raise ValueError('invalid spectral masking setting')
        units = math.ceil(window / mask_block) ** 2
        if mask_mode == 'complementary' and int(round(mask_ratio * 2 * units)) > units:
            raise ValueError('complementary masking needs mask_ratio <= 0.5')
        n = window * window
        self.window, self.n, self.centre, self.units = window, n, n // 2, units
        self.cfg = dict(ch=ch, cl=cl, n_classes=n_classes, window=window, d=d, heads=heads,
                        hidden=hidden, depth=depth, mask_mode=mask_mode, mask_ratio=mask_ratio,
                        mask_split=mask_split, mask_block=mask_block, beta=beta,
                        mask_labeled=bool(mask_labeled), pre_gate=pre_gate,
                        pre_gate_p=float(pre_gate_p), p_spec=float(p_spec), spec_max=float(spec_max),
                        fill='mean', lam=0.0)
        self.stem = nn.ModuleDict({'h': conv_stem(ch, d, d), 'l': conv_stem(cl, d // 2, d)})
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        self.blocks = nn.ModuleList([SFB(d, heads, hidden) for _ in range(depth)])
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.head = nn.Linear(4 * d, n_classes)
        self.branch = True
        self.register_buffer('fill_h', torch.zeros(1, ch, 1, 1))
        self.register_buffer('fill_l', torch.zeros(1, cl, 1, 1))
        # Zero-weight reconstruction heads are retained solely for exact
        # initialization, gradient/optimizer behavior and checkpoint compatibility.
        self.g = nn.ModuleDict({'h': nn.Linear(d, ch), 'l': nn.Linear(d, cl)})
        self.f = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})

    def tokens(self, h, l):
        th = self.stem['h'](h).flatten(2).transpose(1, 2) + self.pos
        tl = self.stem['l'](l).flatten(2).transpose(1, 2) + self.pos
        return th, tl

    def known_lost(self, h, l):
        """pre_gate='known': (B, n) per modality, 1 where the input equals the fill value in every band."""
        return ((h == self.fill_h).all(1).flatten(1).to(h.dtype), (l == self.fill_l).all(1).flatten(1).to(l.dtype))

    def draw_gate(self, batch, g=None, device=None):
        """pre_gate_p < 1: per sample of a masked view, 1 = the pre-gate acts on it, 0 = it does not (B,).
        None when the pre-gate always acts (no random number is drawn then)."""
        if not self.cfg['pre_gate'] or self.cfg['pre_gate_p'] >= 1:
            return None
        return (uniform((batch,), g, device) < self.cfg['pre_gate_p']).float().to(device)

    def key_bias(self, h, l, on=None):
        if self.cfg['pre_gate'] != 'known':
            return None
        kb = {m: LOST_BIAS * x for m, x in zip(MODALITIES, self.known_lost(h, l))}
        if on is not None:
            kb = {m: x * on[:, None] for m, x in kb.items()}
        return kb

    def fuse(self, th, tl, kb=None):
        for b in self.blocks:
            th, tl = b(th, tl, kb)
        return self.norm['h'](th), self.norm['l'](tl)

    def encode(self, h, l, on=None):
        th, tl = self.tokens(h, l)
        return self.fuse(th, tl, self.key_bias(h, l, on))

    def masked_bn(self):
        return bn_frozen(self)

    def pool_tokens(self, z, m):
        return z.mean(1)

    def classify_tokens(self, zh, zl):
        c = self.centre
        return self.head(torch.cat([zh[:, c], zl[:, c], self.pool_tokens(zh, 'h'), self.pool_tokens(zl, 'l')], 1))

    def forward(self, h, l, on=None):
        return self.classify_tokens(*self.encode(h, l, on))

    def set_fill(self, mean_h, mean_l):
        """fill='mean': the per-band values that replace the masked inputs (the band means of the image)."""
        if self.cfg['fill'] != 'mean':
            raise RuntimeError("set_fill is for fill='mean'")
        for buf, v in ((self.fill_h, mean_h), (self.fill_l, mean_l)):
            buf.copy_(torch.as_tensor(v, dtype=buf.dtype).view_as(buf))

    def fill(self, h, l, mh, ml):
        w = self.window
        mh4, ml4 = mh.view(-1, 1, w, w), ml.view(-1, 1, w, w)
        return h * (1 - mh4) + self.fill_h * mh4, l * (1 - ml4) + self.fill_l * ml4

    def draw_masks(self, batch, g=None, device=None):
        c = self.cfg
        if c['mask_block'] == 1:                                          # pixel masks
            return make_masks(batch, self.n, c['mask_ratio'], c['mask_mode'], c['mask_split'], g, device)
        mh, ml = make_masks(batch, self.units, c['mask_ratio'], c['mask_mode'], c['mask_split'], g, device)
        idx = block_index(self.window, c['mask_block'], mh.device)      # block masks, spread to the pixels
        return mh[:, idx], ml[:, idx]

    def masked_inputs(self, h, l, g=None):
        """Spatial masking of the labeled counterpart only."""
        mh, ml = self.draw_masks(h.shape[0], g, h.device)
        return self.fill(h, l, mh, ml)

    def pool_losses(self, h, l, g=None, masks=None, on=None, *, spec_g=None):
        """M2C: spectral masking only on spatially visible HSI student inputs."""
        if not self.branch:
            raise RuntimeError('The training branch has been removed')
        if spec_g is None:
            if g is None:
                raise ValueError('Provide the independent CPU spectral generator or the spatial generator')
            seed = int(g.initial_seed()) - 1_000_003
            expected = seed + 3_000_003
            spec_g = self.__dict__.get('_spec_generator')
            if spec_g is None:
                spec_g = torch.Generator().manual_seed(expected)
                self.__dict__['_spec_generator'] = spec_g
            elif spec_g.initial_seed() != expected:
                raise RuntimeError('Spectral generator belongs to a different seed')
        mh, ml = masks if masks is not None else self.draw_masks(h.shape[0], g, h.device)
        bh = draw_spectral(h, spec_g, self.cfg['spec_max'], self.cfg['p_spec'])
        out = {'mh': mh, 'ml': ml}
        with self.masked_bn():
            hm, lm = self.fill(h, l, mh, ml)
            zh, zl = self.encode(apply_spectral(hm, mh, bh, self.fill_h), lm, on)
            xh, xl = h.flatten(2).transpose(1, 2), l.flatten(2).transpose(1, 2)
            mse_h = masked_mean(((self.g['h'](zh) - xh) ** 2).mean(2), mh)
            mse_l = masked_mean(((self.g['l'](zl) - xl) ** 2).mean(2), ml)
            out['rec'] = mse_h + mse_l
            out['mse_h'], out['mse_l'] = mse_h.detach(), mse_l.detach()
            with torch.no_grad():
                ch_, cl_ = self.encode(h, l)
            ph, pl = self.f['h'](zh), self.f['l'](zl)
            out['feat'] = (masked_mean(((ph - ch_) ** 2).mean(2), mh)
                           + masked_mean(((pl - cl_) ** 2).mean(2), ml))
        return out

    def is_test_time(self, name):
        return name.startswith(TEST_TIME_PREFIXES)

    def count_parameters(self):
        test = sum(p.numel() for k, p in self.named_parameters() if self.is_test_time(k))
        return dict(test_time=test, training_only=sum(p.numel() for p in self.parameters()) - test)

    def strip_training_branch(self):
        """Remove training-only heads; keep the mean-fill buffers for GSFB."""
        for k in ('g', 'f'):
            if hasattr(self, k):
                delattr(self, k)
        self.branch = False
        return self
