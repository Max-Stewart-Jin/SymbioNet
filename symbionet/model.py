"""SymbioNet: Symbiotic Cross-Modal Completion Network for hyperspectral (HSI) and LiDAR joint classification.

Deployed network (test time):  per-modality conv stems + shared positional embedding P -> symmetric fusion encoder E
                               (two inputs, two outputs) -> linear classifier.
Training only:                 per-modality (complementary) masking with learnable fill values, a dual-input decoder D
                               (bidirectional Mamba or symmetric transformer blocks), linear reconstruction heads g_H,
                               g_L, a cross-path feature target (the encoder tokens of the unmasked patch) and a
                               weighted masked-reconstruction loss.
strip_training_branch() deletes everything training-only; the classifier's output does not change.

Notation: h = HSI patch (B, ch, w, w), l = LiDAR patch (B, cl, w, w); n = w * w tokens of width d; m in {h, l}.

Symmetric fusion block (SFB; every modality has its own LN, Q / K / V / O and MLP):
    Q_m, K_m, V_m = LN_m(T_m) W^m_{Q,K,V}
    T_m <- T_m + softmax(Q_m [K_h ; K_l]^T / sqrt(d_head)) [V_h ; V_l] W^m_O      (per head, over all 2n tokens)
    T_m <- T_m + MLP_m(LN'_m(T_m))
Encoder E: T_m = Stem_m(x_m) + P, `depth` SFBs, Z_m = LN_m(T_m).
Classifier: logits = W [z_h,c ; z_l,c ; mean(Z_h) ; mean(Z_l)] + b   (c = centre token).

Masking (training branch, per pool patch): round(ratio * 2n) of the 2n (position, modality) slots are masked; the HSI
share is U(0, 1) (Dirichlet(1, 1)) unless mask_split='equal'. mask_mode:
    'independent'    positions drawn separately for the two modalities
    'same'           both modalities masked at the same round(ratio * n) positions
    'complementary'  no position masked in both (M_l = 1 - M_h at ratio 0.5): the other modality is always visible
mask_block = b > 1: the unit is a b x b block instead of a pixel. The patch is tiled from its top-left corner into
u = ceil(w / b)^2 blocks (edge blocks are smaller when b does not divide w), the masks above are drawn over the u blocks
and every pixel takes the mask of its block, so a hidden pixel's neighbours of the same modality are hidden too.
b = w hides a whole modality of the patch.
Masked inputs are replaced by learnable fill values t_m. Pool batches never update the BatchNorm running statistics.

Decoder D (decoder=...):
    None            no training branch (the base network)
    'identity'      heads read Z directly (no decoder)
    'transformer'   dec_depth SFBs + LN
    'mamba'         dec_depth bidirectional Mamba blocks over the 2n tokens interleaved by position
                    (h_1, l_1, h_2, l_2, ...) with a learnable modality embedding; plain-PyTorch selective scan
Losses on a pool batch (masked slots only; squared errors averaged over channels):
    MSE_m  = mean_{p in M_m} ||g_m(U_m')_p - x_m,p||^2 / c_m      rec_source 'self':  m' = m
                                                                  rec_source 'cross': m' = the other modality
             ('cross': the hidden value of one modality at p is predicted from the token of the other modality at
             the same position p; with complementary masking that modality is always visible there. L_feat is the
             same for both.)
    L_rec  = rec_weighting 'kendall':  sum_m [exp(-s_m) MSE_m + s_m]       (learned s_m = log sigma_m^2)
                           'balanced': exp(-s) MSE_h + exp(s) MSE_l         (weights multiply to 1, learned s)
                           'none':     MSE_h + MSE_l
             (all three equal MSE_h + MSE_l at initialisation, s = 0)
    L_feat = sum_m mean_{p in M_m} ||f_m(U_m)_p - sg(Z^clean_m,p)||^2 / d   (feat=True)
    L_cons = mean KL(sg(p_clean) || p_masked) over pool patches with clean confidence >= cons_tau (cons=True)
Total: L = L_CE + lam * L_rec + beta * L_feat + gamma * L_cons.

Options of the deployed network (the defaults give the network above; each option is tested on its own):
    stem='spectral'  HSI stem = a 1-D convolution along the bands shared by all pixels (8 filters of length 7, stride 2)
                     + BN + GELU, a 1x1 projection of the spectral features to d + BN + GELU, then two depthwise-
                     separable 3x3 layers (depthwise 3x3, pointwise 1x1, BN, GELU). Receptive field 5 x 5 as the conv
                     stem, fewer parameters. The LiDAR stem is unchanged.
    rel_bias=True    every SFB adds a learned relative position bias to its attention scores, one table per head and
                     per (query modality, key modality) pair:
                         s_(m,p),(m',p') = q_m,p . k_m',p' / sqrt(d_head) + B^head_{m->m'}[p' - p]
                     (zero at initialisation; B_{h->l}[0] is the HSI token reading the LiDAR token at its own position)
    readout='attn'   the classifier reads the centre tokens and a centre-guided attention pooling of each modality
                         z~_m = sum_p a_m,p z_m,p,   a_m = softmax_p((W_q z_m,c) . (W_k z_m,p) / sqrt(16))
                     instead of the mean tokens (W_q = 0 at initialisation, which is exactly the mean)
    fusion=...       ablations of the fusion block in the encoder (the decoder keeps the SFB):
                     'joint'     the SFB above: every query attends to the 2n keys [K_h ; K_l] with one softmax
                     'self'      every query attends to the keys of its own modality only (no exchange inside the
                                 encoder; the classifier still reads both modalities)
                     'to_hsi'    HSI queries attend to [K_h ; K_l], LiDAR queries to K_l only (one-way, LiDAR -> HSI)
                     'to_lidar'  LiDAR queries attend to [K_h ; K_l], HSI queries to K_h only (one-way, HSI -> LiDAR)
                     'shared'    'joint' with one LN / Q / K / V / O / MLP for both modalities, plus a learned modality
                                 embedding added to the tokens (self-attention over the concatenated tokens)
                     'fixed'     two separate softmaxes, over the own and over the other modality's keys, mixed with
                                 the fixed weights 1/2 (the SFB mixes the same two terms with the adaptive weight alpha)

Options for lost data (tested on their own; the defaults give the network above):
    fill='mean'        the masked inputs are replaced by the band means of the image (fixed, set_fill()) instead of
                       learned values: the values that replace lost pixels when data are imputed with the band means,
                       so a lost pixel at test time looks like a masked pixel in training
    feat_head='none'   L_feat compares the encoder tokens themselves with the clean tokens, without the heads f_m:
                           L_feat = sum_m mean_{p in M_m} ||z~_m,p - sg(z_m,p)||^2 / d
    gate=True          gated completion, also at test time (needs decoder='identity', feat=True, feat_head='linear'):
                           r_m,p = sigmoid(w_m . z_m,p + b_m),      z'_m,p = (1 - r_m,p) z_m,p + r_m,p f_m(z_m,p)
                       and the classifier reads z'. The gate is trained to detect the masked slots,
                           L_gate = 1/2 sum_m [BCE(r(z~_m), M_m) + BCE(r(sg(z_m)), 0)]    (masked view, clean view)
                       and no other loss passes a gradient through r. The gate starts closed (w = 0, b = -4), and
                       with intact data r ~ 0 and z' ~ z.
    gate='known'       the same completion with r = 1 where the input of the slot equals the fill value in every band
                       and 0 elsewhere (needs fill='mean'; no gate parameters, no L_gate): at a lost slot the
                       classifier reads the completed feature f_m(z_m,p), elsewhere z_m,p
    mask_labeled=True  the classification loss is also computed on a masked copy of every labeled batch (the plain
                       augmentation baseline; works without the training branch when fill='mean')
    pre_gate=...       reliability gate before the fusion blocks, also at test time: r_m,p in [0, 1] says how likely
                       the slot (m, p) is lost, and every SFB adds log(1 - r_m',p') to the attention scores of the key
                       (m', p'), so a softmax weight is multiplied by 1 - r of its key and renormalised. Lost slots
                       are not read by the other tokens (a lost modality: the other one attends to itself only);
                       with intact data it is the SFB.
                       'known'    r = 1 where the input of the slot equals the fill value in every band, else 0
                                  (needs fill='mean': lost values imputed with the band means are recognised exactly;
                                  no parameters, no extra loss)
                       'learned'  r_m,p = sigmoid(w_m . t_m,p + b_m) from the stem tokens T_m, before any exchange
                                  between the modalities (needs the training branch and feat=True). The gate is
                                  trained to detect the masked slots,
                                      L_pre = 1/2 sum_m [BCE(r(t~_m), M_m) + BCE(r(sg(t_m)), 0)]   (masked, clean view)
                                  and no other loss passes a gradient through r. It starts closed (w = 0, b = -4):
                                  the same value for every key, which leaves the softmax unchanged.
    band_mask=b        b > 0: every masked view (a pool patch, a masked labeled copy) also loses HSI bands: a share
                       U(0, b) of the bands, drawn per sample, is replaced by the fill values at every position of the
                       patch (draw_bands()), on top of the masked positions. The losses are unchanged (L_feat is still
                       taken at the masked positions). 0 (default): positions only.
    pre_gate_p=p       training only, p < 1: the pre-gate acts on a masked view (a pool patch, a masked labeled copy)
                       with probability p, drawn per sample (draw_gate()); in the other masked views the lost slots are
                       read like any other slot, so the network also learns to cope with bad tokens that no gate
                       points out. At test time the pre-gate always acts. p = 1 (default): it always acts.
                       (Such a model can be tested without the gate: robustness.py --set model.pre_gate=false.)
    bn_masked='running'  the masked views (pool batches, masked labeled copies) are normalised by the BatchNorm layers
                       with the running statistics, as every input is at test time, instead of with the statistics of
                       their own batch ('batch'); the running statistics are updated by the clean labeled batches only
                       in both cases
"""
import math
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ('h', 'l')
MASK_MODES = ('independent', 'same', 'complementary')
DECODERS = (None, 'identity', 'transformer', 'mamba')
REC_WEIGHTINGS = ('kendall', 'balanced', 'none')
REC_SOURCES = ('self', 'cross')
STEMS = ('conv', 'spectral')
READOUTS = ('mean', 'attn')
FUSIONS = ('joint', 'self', 'to_hsi', 'to_lidar', 'shared', 'fixed')
FILLS = ('learned', 'mean')
FEAT_HEADS = ('linear', 'none')
BN_MASKED = ('batch', 'running')
PRE_GATES = (False, 'known', 'learned')
GATES = (False, True, 'known')
LOST_BIAS = -30.0            # log(1 - r) of a slot known to be lost (a softmax weight times 1e-13)
READS_OTHER = {'joint': 'hl', 'self': '', 'to_hsi': 'h', 'to_lidar': 'l', 'shared': 'hl'}   # queries that see both
TEST_TIME_PREFIXES = ('stem.', 'pos', 'mod_emb', 'blocks.', 'norm.', 'head.', 'readout.')


# ---------------------------------------------------------------------------------------- PyTorch check (CPU)
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
    """Two 3x3 conv + BN + GELU layers (the stems of the JD base)."""
    return nn.Sequential(nn.Conv2d(cin, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
                         nn.Conv2d(mid, d, 3, padding=1), nn.BatchNorm2d(d), nn.GELU())


def dsconv(d):
    """Depthwise-separable 3x3 layer: depthwise 3x3, pointwise 1x1, BN, GELU."""
    return nn.Sequential(nn.Conv2d(d, d, 3, padding=1, groups=d), nn.Conv2d(d, d, 1), nn.BatchNorm2d(d), nn.GELU())


class SpectralStem(nn.Module):
    """HSI stem (stem='spectral'): a 1-D convolution along the bands shared by all pixels (s filters of length k,
    stride r) + BN + GELU, a 1x1 projection of the s * ceil(C / r) spectral features of each pixel to d + BN + GELU,
    then two depthwise-separable 3x3 layers. Receptive field 5 x 5, as conv_stem."""

    def __init__(self, cin, d, s=8, k=7, r=2):
        super().__init__()
        self.spec = nn.Sequential(nn.Conv3d(1, s, (k, 1, 1), stride=(r, 1, 1), padding=(k // 2, 0, 0)),
                                  nn.BatchNorm3d(s), nn.GELU())
        bands = (cin + 2 * (k // 2) - k) // r + 1                          # length of the spectral axis after spec
        self.proj = nn.Sequential(nn.Conv2d(s * bands, d, 1), nn.BatchNorm2d(d), nn.GELU())
        self.spat = nn.Sequential(dsconv(d), dsconv(d))

    def forward(self, x):                                                  # x: (B, C, w, w)
        y = self.spec(x.unsqueeze(1))                                      # (B, s, bands, w, w)
        return self.spat(self.proj(y.flatten(1, 2)))                       # (B, d, w, w)


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


def relative_index(window):
    """Index of the offset p' - p of every pair of positions of a window x window patch, in [0, (2w - 1)^2)."""
    r = torch.arange(window)
    rows, cols = [a.flatten() for a in torch.meshgrid(r, r, indexing='ij')]
    dr = rows[None, :] - rows[:, None] + window - 1                        # [query p, key p']
    dc = cols[None, :] - cols[:, None] + window - 1
    return dr * (2 * window - 1) + dc


def relative_onehot(window):
    """(R, n * n) one-hot matrix of relative_index: table (.., R) @ onehot gives the bias of every pair. A matrix
    product keeps the gradient of the tables deterministic on the GPU (indexing would use atomic adds)."""
    idx = relative_index(window).flatten()
    oh = torch.zeros((2 * window - 1) ** 2, idx.numel())
    oh[idx, torch.arange(idx.numel())] = 1.0
    return oh


REL_PAIRS = ('hh', 'hl', 'lh', 'll')                                       # (query modality, key modality)


class SFB(nn.Module):
    """Two inputs, two outputs: each modality queries the tokens of both modalities with its own Q / K / V.
    rel_window = w adds a learned relative position bias per head and per (query, key) modality pair.
    fusion != 'joint' gives the ablations of the block listed in the module docstring."""

    def __init__(self, d=64, heads=4, hidden=128, rel_window=None, fusion='joint'):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by heads')
        if fusion not in FUSIONS:
            raise ValueError('fusion must be one of %s' % (FUSIONS,))
        self.heads, self.fusion = heads, fusion
        sets = ('h',) if fusion == 'shared' else MODALITIES                # parameter sets ('shared': one for both)
        self.n1 = nn.ModuleDict({m: nn.LayerNorm(d) for m in sets})
        self.qkv = nn.ModuleDict({m: nn.Linear(d, 3 * d) for m in sets})
        self.o = nn.ModuleDict({m: nn.Linear(d, d) for m in sets})
        self.n2 = nn.ModuleDict({m: nn.LayerNorm(d) for m in sets})
        self.mlp = nn.ModuleDict({m: mlp(d, hidden) for m in sets})
        self.rel = None
        if rel_window is not None:                                         # zero: the plain SFB at initialisation
            R = (2 * rel_window - 1) ** 2
            self.rel = nn.ParameterDict({p: nn.Parameter(torch.zeros(heads, R)) for p in REL_PAIRS})

    def rel_blocks(self, onehot):
        """{(query modality, key modality): (heads, n, n)} relative position bias of every modality pair."""
        n = int(round(math.sqrt(onehot.shape[1])))
        e = (torch.cat([self.rel[p] for p in REL_PAIRS], 0) @ onehot).view(4, self.heads, n, n)
        return {(p[0], p[1]): e[i] for i, p in enumerate(REL_PAIRS)}

    def rel_bias(self, onehot):
        """{query modality: (heads, n, 2n) bias over the keys [K_h ; K_l]}."""
        b = self.rel_blocks(onehot)
        return {m: torch.cat([b[(m, 'h')], b[(m, 'l')]], -1) for m in MODALITIES}

    def attend(self, m, q, k, v, b, kb=None):
        """Attention output of the queries of modality m, before W_O (b: rel_blocks or None; kb: None or
        {modality: (B, n)}, added to the scores of the keys of that modality)."""
        o = 'l' if m == 'h' else 'h'

        def bias(a, c):
            return None if b is None else b[(a, c)]

        def kbias(c):
            return None if kb is None else kb[c]
        if self.fusion == 'fixed':
            return 0.5 * (attention(q[m], k[m], v[m], self.heads, bias(m, m), kbias(m))
                          + attention(q[m], k[o], v[o], self.heads, bias(m, o), kbias(o)))
        if m not in READS_OTHER[self.fusion]:                              # the keys of its own modality only
            return attention(q[m], k[m], v[m], self.heads, bias(m, m), kbias(m))
        K, V = torch.cat([k['h'], k['l']], 1), torch.cat([v['h'], v['l']], 1)
        B = None if b is None else torch.cat([b[(m, 'h')], b[(m, 'l')]], -1)
        return attention(q[m], K, V, self.heads, B, None if kb is None else torch.cat([kb['h'], kb['l']], 1))

    def forward(self, th, tl, onehot=None, kb=None):
        t = {'h': th, 'l': tl}
        p = {m: ('h' if self.fusion == 'shared' else m) for m in MODALITIES}    # parameter set used by modality m
        q, k, v = {}, {}, {}
        for m in MODALITIES:
            q[m], k[m], v[m] = self.qkv[p[m]](self.n1[p[m]](t[m])).chunk(3, -1)
        b = self.rel_blocks(onehot) if self.rel is not None else None
        out = []
        for m in MODALITIES:
            x = t[m] + self.o[p[m]](self.attend(m, q, k, v, b, kb))
            out.append(x + self.mlp[p[m]](self.n2[p[m]](x)))
        return out[0], out[1]


# ---------------------------------------------------------------------------------------- selective SSM (Mamba)
def _run_forward(a, b):
    """h_t = a_t * h_{t-1} + b_t from h_{-1} = 0, along dim 0 (time-major (L, B, c, S)); returns every h_t."""
    h = torch.empty_like(b)
    hs, as_, bs = h.unbind(0), a.unbind(0), b.unbind(0)           # per-step views, made once
    hs[0].copy_(bs[0])
    for t in range(1, len(hs)):
        torch.addcmul(bs[t], as_[t], hs[t - 1], out=hs[t])
    return h


def _run_backward(a, g):
    """In place: g_t <- g_t + a_{t+1} * g_{t+1}, from the last step back (the adjoint of _run_forward)."""
    gs, as_ = g.unbind(0), a.unbind(0)
    for t in range(len(gs) - 2, -1, -1):
        torch.addcmul(gs[t], as_[t + 1], gs[t + 1], out=gs[t])
    return g


def _time_major(*xs):
    return [x.transpose(0, 1).contiguous() for x in xs]


class SelectiveScan(torch.autograd.Function):
    """y_t = sum_s C_t,s h_t,s,  h_t = exp(dt_t A) * h_{t-1} + dt_t u_t B_t,  h_0 = 0  (diagonal A per channel).

    u, dt: (B, L, c) with dt > 0; A: (c, S) with A < 0; Bm, Cm: (B, L, S). Plain PyTorch, one fused multiply-add
    per step. Only the inputs are kept for the backward pass, which recomputes the states h and then runs the same
    recurrence backwards in time (the adjoint): g_t = dL/dh_t = gy_t C_t + exp(dt_{t+1} A) * g_{t+1}. So the memory
    does not grow with B * L * c * S, and there is no dependence on torch.utils.checkpoint.
    """

    @staticmethod
    def forward(ctx, u, dt, A, Bm, Cm):
        uT, dtT, BT, CT = _time_major(u, dt, Bm, Cm)
        H = _run_forward(torch.exp(dtT.unsqueeze(-1) * A), (dtT * uT).unsqueeze(-1) * BT.unsqueeze(2))
        ctx.save_for_backward(u, dt, A, Bm, Cm)
        return (H * CT.unsqueeze(2)).sum(-1).transpose(0, 1).contiguous()

    @staticmethod
    def backward(ctx, gy):
        u, dt, A, Bm, Cm = ctx.saved_tensors
        uT, dtT, BT, CT, gyT = _time_major(u, dt, Bm, Cm, gy)
        dA = torch.exp(dtT.unsqueeze(-1) * A)                    # a_t, (L, B, c, S)
        dtu = dtT * uT
        H = _run_forward(dA, dtu.unsqueeze(-1) * BT.unsqueeze(2))
        gC = (gyT.unsqueeze(-1) * H).sum(2)                      # dL/dC_t
        G = _run_backward(dA, gyT.unsqueeze(-1) * CT.unsqueeze(2))   # dL/dh_t = dL/db_t (b_t = dt_t u_t B_t)
        gB = (G * dtu.unsqueeze(-1)).sum(2)
        gdtu = (G * BT.unsqueeze(2)).sum(-1)
        G.mul_(dA)                                               # dL/d(dt_t A) = dL/da_t * a_t, with
        G[0] = 0                                                 # dL/da_t = g_t * h_{t-1} and h_{-1} = 0
        G[1:].mul_(H[:-1])
        gdt = (G * A).sum(-1) + gdtu * uT
        gA = (G * dtT.unsqueeze(-1)).sum((0, 1))
        back = lambda x: x.transpose(0, 1).contiguous()          # noqa: E731
        return back(gdtu * dtT), back(gdt), gA, back(gB), back(gC)


def selective_scan(u, dt, A, Bm, Cm, D):
    """y_t = C_t . h_t + D * u_t (see SelectiveScan)."""
    return SelectiveScan.apply(u, dt, A, Bm, Cm) + u * D


class MambaMixer(nn.Module):
    """Mamba-1 selective-SSM mixer (Gu & Dao, 2023) in plain PyTorch: in_proj -> causal depthwise conv -> SiLU ->
    input-dependent (dt, B, C) -> selective scan -> gated by SiLU(z) -> out_proj."""

    def __init__(self, d, d_state=8, expand=1, d_conv=4, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        c = expand * d
        self.c, self.S, self.R = c, d_state, max(1, math.ceil(d / 16))
        self.in_proj = nn.Linear(d, 2 * c)
        self.conv = nn.Conv1d(c, c, d_conv, groups=c, padding=d_conv - 1)
        self.x_proj = nn.Linear(c, self.R + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.R, c)
        dt = torch.exp(torch.rand(c) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))       # softplus(bias) = dt
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(c, 1))
        self.D = nn.Parameter(torch.ones(c))
        self.out_proj = nn.Linear(c, d)

    def forward(self, x):
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt, Bm, Cm = self.x_proj(u).split([self.R, self.S, self.S], -1)
        y = selective_scan(u, F.softplus(self.dt_proj(dt)), -torch.exp(self.A_log), Bm, Cm, self.D)
        return self.out_proj(y * F.silu(z))


class BiMambaBlock(nn.Module):
    """Pre-LN residual block: forward and backward scans of the sequence, then an MLP."""

    def __init__(self, d, hidden, **kw):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fwd, self.bwd = MambaMixer(d, **kw), MambaMixer(d, **kw)
        self.mlp = mlp(d, hidden)

    def forward(self, x):
        y = self.n1(x)
        x = x + self.fwd(y) + self.bwd(y.flip(1)).flip(1)
        return x + self.mlp(self.n2(x))


# --------------------------------------------------------------------------------------------------- decoders
class TransformerDecoder(nn.Module):
    def __init__(self, d, heads, hidden, depth):
        super().__init__()
        self.blocks = nn.ModuleList([SFB(d, heads, hidden) for _ in range(depth)])
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})

    def forward(self, zh, zl):
        for b in self.blocks:
            zh, zl = b(zh, zl)
        return self.norm['h'](zh), self.norm['l'](zl)


class MambaDecoder(nn.Module):
    """Tokens interleaved by position (h_1, l_1, ..., h_n, l_n) plus a learnable modality embedding."""

    def __init__(self, d, hidden, depth, **kw):
        super().__init__()
        self.mod = nn.Parameter(torch.zeros(2, d))
        self.blocks = nn.ModuleList([BiMambaBlock(d, hidden, **kw) for _ in range(depth)])
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})

    def forward(self, zh, zl):
        B, n, d = zh.shape
        x = torch.stack([zh + self.mod[0], zl + self.mod[1]], 2).reshape(B, 2 * n, d)
        for b in self.blocks:
            x = b(x)
        x = x.reshape(B, n, 2, d)
        return self.norm['h'](x[:, :, 0]), self.norm['l'](x[:, :, 1])


# ---------------------------------------------------------------------------------------------------- masking
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
    """1 = masked; float masks (B, n) for the HSI and the LiDAR input (see the top of the file)."""
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


@contextmanager
def bn_running(net):
    """BatchNorm layers normalise with their running statistics, as at test time, and do not update them."""
    bns = [x for x in net.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm) and x.training]
    for x in bns:
        x.eval()
    try:
        yield
    finally:
        for x in bns:
            x.train()


# ------------------------------------------------------------------------------------------------------ model
class SymbioNet(nn.Module):
    def __init__(self, ch, cl, n_classes, window=11, d=64, heads=4, hidden=128, depth=2,
                 decoder='transformer', dec_depth=2, d_state=8, expand=1,
                 mask_mode='independent', mask_ratio=0.5, mask_split='dirichlet', mask_block=1,
                 rec_weighting='kendall', rec_source='self', feat=False, cons=False, cons_tau=0.0, lam=1.0, beta=1.0,
                 gamma=1.0, stem='conv', rel_bias=False, readout='mean', fusion='joint', fill='learned',
                 feat_head='linear', gate=False, mask_labeled=False, pre_gate=False, bn_masked='batch',
                 pre_gate_p=1.0, band_mask=0.0):
        super().__init__()
        if stem not in STEMS:
            raise ValueError('stem must be one of %s' % (STEMS,))
        if readout not in READOUTS:
            raise ValueError('readout must be one of %s' % (READOUTS,))
        if fusion not in FUSIONS:
            raise ValueError('fusion must be one of %s' % (FUSIONS,))
        if fill not in FILLS:
            raise ValueError('fill must be one of %s' % (FILLS,))
        if feat_head not in FEAT_HEADS:
            raise ValueError('feat_head must be one of %s' % (FEAT_HEADS,))
        gate = gate or False
        if not any(gate is x or (isinstance(gate, str) and gate == x) for x in GATES):
            raise ValueError('gate must be one of %s' % (GATES,))
        if gate and not (decoder == 'identity' and feat and feat_head == 'linear'):
            raise ValueError("gate needs decoder='identity', feat=True and feat_head='linear'")
        if gate == 'known' and fill != 'mean':
            raise ValueError("gate='known' needs fill='mean'")
        if mask_labeled and decoder is None and fill != 'mean':
            raise ValueError("mask_labeled without the training branch needs fill='mean'")
        pre_gate = pre_gate or False
        if pre_gate not in PRE_GATES:
            raise ValueError('pre_gate must be one of %s' % (PRE_GATES,))
        if pre_gate == 'learned' and (decoder is None or not feat):
            raise ValueError("pre_gate='learned' needs the training branch (a decoder) and feat=True")
        if pre_gate == 'known' and fill != 'mean':
            raise ValueError("pre_gate='known' needs fill='mean'")
        if isinstance(pre_gate_p, bool) or not isinstance(pre_gate_p, (int, float)) or not 0 < pre_gate_p <= 1:
            raise ValueError('pre_gate_p must be a probability in (0, 1]')         # unused without a pre-gate
        if isinstance(band_mask, bool) or not isinstance(band_mask, (int, float)) or not 0 <= band_mask <= 1:
            raise ValueError('band_mask must be a share of the HSI bands in [0, 1]')
        if bn_masked not in BN_MASKED:
            raise ValueError('bn_masked must be one of %s' % (BN_MASKED,))
        if decoder not in DECODERS:
            raise ValueError('decoder must be one of %s' % (DECODERS,))
        if rec_weighting not in REC_WEIGHTINGS:
            raise ValueError('rec_weighting must be one of %s' % (REC_WEIGHTINGS,))
        if rec_source not in REC_SOURCES:
            raise ValueError('rec_source must be one of %s' % (REC_SOURCES,))
        if mask_mode not in MASK_MODES:
            raise ValueError('mask_mode must be one of %s' % (MASK_MODES,))
        if not (isinstance(mask_block, int) and 1 <= mask_block <= window):
            raise ValueError('mask_block must be an integer from 1 to the patch size %d' % window)
        units = math.ceil(window / mask_block) ** 2                     # masking units: pixels or blocks
        if mask_mode == 'complementary' and int(round(mask_ratio * 2 * units)) > units:
            raise ValueError('complementary masking needs mask_ratio <= 0.5')
        n = window * window
        self.window, self.n, self.centre, self.units = window, n, n // 2, units
        self.cfg = dict(ch=ch, cl=cl, n_classes=n_classes, window=window, d=d, heads=heads, hidden=hidden,
                        depth=depth, decoder=decoder, dec_depth=dec_depth, d_state=d_state, expand=expand,
                        mask_mode=mask_mode, mask_ratio=mask_ratio, mask_split=mask_split, mask_block=mask_block,
                        rec_weighting=rec_weighting, rec_source=rec_source, feat=feat, cons=cons, cons_tau=cons_tau,
                        lam=lam, beta=beta, gamma=gamma, stem=stem, rel_bias=bool(rel_bias), readout=readout,
                        fusion=fusion, fill=fill, feat_head=feat_head, gate=gate,
                        mask_labeled=bool(mask_labeled), pre_gate=pre_gate, bn_masked=bn_masked,
                        pre_gate_p=float(pre_gate_p), band_mask=float(band_mask))
        # deployed part
        hsi_stem = conv_stem(ch, d, d) if stem == 'conv' else SpectralStem(ch, d)
        self.stem = nn.ModuleDict({'h': hsi_stem, 'l': conv_stem(cl, d // 2, d)})
        self.pos = nn.Parameter(torch.zeros(1, n, d))
        if fusion == 'shared':                                            # tells the shared projections the modality
            self.mod_emb = nn.Parameter(torch.zeros(2, d))
        self.blocks = nn.ModuleList([SFB(d, heads, hidden, rel_window=window if rel_bias else None, fusion=fusion)
                                     for _ in range(depth)])
        if rel_bias:                                                      # rebuilt from the window, not saved
            self.register_buffer('rel_onehot', relative_onehot(window), persistent=False)
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.head = nn.Linear(4 * d, n_classes)
        self.pgate = None
        if pre_gate == 'learned':                                         # starts closed: r = sigmoid(-4) ~ 0.02
            self.pgate = nn.ModuleDict({m: nn.Linear(d, 1) for m in MODALITIES})
            for m in MODALITIES:
                nn.init.zeros_(self.pgate[m].weight)
                nn.init.constant_(self.pgate[m].bias, -4.0)
        if readout == 'attn':                                             # W_q = 0: exactly the mean at initialisation
            self.readout = nn.ModuleDict({m: nn.ModuleDict({'q': nn.Linear(d, 16), 'k': nn.Linear(d, 16)})
                                          for m in MODALITIES})
            for m in MODALITIES:
                nn.init.zeros_(self.readout[m]['q'].weight)
                nn.init.zeros_(self.readout[m]['q'].bias)
        # training branch
        self.branch = decoder is not None
        self.dec = self.f = self.gate = None
        if fill == 'mean':                                                # fixed band means, given by set_fill()
            self.register_buffer('fill_h', torch.zeros(1, ch, 1, 1))
            self.register_buffer('fill_l', torch.zeros(1, cl, 1, 1))
        if self.branch:
            if fill == 'learned':
                self.fill_h = nn.Parameter(torch.zeros(1, ch, 1, 1))
                self.fill_l = nn.Parameter(torch.zeros(1, cl, 1, 1))
            if decoder == 'transformer':
                self.dec = TransformerDecoder(d, heads, hidden, dec_depth)
            elif decoder == 'mamba':
                self.dec = MambaDecoder(d, hidden, dec_depth, d_state=d_state, expand=expand)
            self.g = nn.ModuleDict({'h': nn.Linear(d, ch), 'l': nn.Linear(d, cl)})
            n_s = {'kendall': 2, 'balanced': 1, 'none': 0}[rec_weighting]
            self.s = nn.Parameter(torch.zeros(n_s)) if n_s else None
            if feat and feat_head == 'linear':
                self.f = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
            if gate is True:                                              # starts closed: r = sigmoid(-4) ~ 0.02
                self.gate = nn.ModuleDict({m: nn.Linear(d, 1) for m in MODALITIES})
                for m in MODALITIES:
                    nn.init.zeros_(self.gate[m].weight)
                    nn.init.constant_(self.gate[m].bias, -4.0)

    # ---- deployed path
    def tokens(self, h, l):
        """The stem tokens T_h, T_l (with the positional embedding), before the fusion blocks."""
        th = self.stem['h'](h).flatten(2).transpose(1, 2) + self.pos
        tl = self.stem['l'](l).flatten(2).transpose(1, 2) + self.pos
        if self.cfg['fusion'] == 'shared':
            th, tl = th + self.mod_emb[0], tl + self.mod_emb[1]
        return th, tl

    def lost_logits(self, th, tl):
        """pre_gate='learned': the logits of r, (B, n) per modality (r = the probability that the slot is lost)."""
        return self.pgate['h'](th).squeeze(-1), self.pgate['l'](tl).squeeze(-1)

    def known_lost(self, h, l):
        """pre_gate='known': (B, n) per modality, 1 where the input equals the fill value in every band."""
        return ((h == self.fill_h).all(1).flatten(1).to(h.dtype), (l == self.fill_l).all(1).flatten(1).to(l.dtype))

    def draw_gate(self, batch, g=None, device=None):
        """pre_gate_p < 1: per sample of a masked view, 1 = the pre-gate acts on it, 0 = it does not (B,).
        None when the pre-gate always acts (no random number is drawn then)."""
        if not self.cfg['pre_gate'] or self.cfg['pre_gate_p'] >= 1:
            return None
        return (uniform((batch,), g, device) < self.cfg['pre_gate_p']).float().to(device)

    def key_bias(self, h, l, th, tl, on=None):
        """log(1 - r) of every slot, {modality: (B, n)}, or None without a pre-gate (no gradient through r);
        with the learned gate also its logits (s_h, s_l). on: None, or (B,) with 0 for the samples whose keys get
        no bias (draw_gate())."""
        if self.cfg['pre_gate'] == 'known':
            kb = {m: LOST_BIAS * x for m, x in zip(MODALITIES, self.known_lost(h, l))}
            logits = None
        elif self.cfg['pre_gate'] == 'learned':
            logits = self.lost_logits(th, tl)
            kb = {m: F.logsigmoid(-s).detach() for m, s in zip(MODALITIES, logits)}
        else:
            return None, None
        if on is not None:
            kb = {m: x * on[:, None] for m, x in kb.items()}
        return kb, logits

    def fuse(self, th, tl, kb=None):
        """The fusion blocks and the final norms; kb: the key bias of the pre-gate, added in every block."""
        oh = self.rel_onehot if self.cfg['rel_bias'] else None
        for b in self.blocks:
            th, tl = b(th, tl, oh) if kb is None else b(th, tl, oh, kb)
        return self.norm['h'](th), self.norm['l'](tl)

    def encode(self, h, l, on=None):
        th, tl = self.tokens(h, l)
        return self.fuse(th, tl, self.key_bias(h, l, th, tl, on)[0])

    def masked_bn(self):
        """Context of the masked views: how their BatchNorm layers normalise (bn_masked)."""
        return bn_running(self) if self.cfg['bn_masked'] == 'running' else bn_frozen(self)

    def pool_tokens(self, z, m):
        """Summary token of modality m: the mean, or the centre-guided attention pooling (readout='attn')."""
        if self.cfg['readout'] == 'mean':
            return z.mean(1)
        q = self.readout[m]['q'](z[:, self.centre])                       # (B, 16)
        k = self.readout[m]['k'](z)                                       # (B, n, 16)
        a = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / 4.0, 1)    # / sqrt(16)
        return (a.unsqueeze(1) @ z).squeeze(1)

    def classify_tokens(self, zh, zl):
        c = self.centre
        return self.head(torch.cat([zh[:, c], zl[:, c], self.pool_tokens(zh, 'h'), self.pool_tokens(zl, 'l')], 1))

    def complete_tokens(self, zh, zl, lost=None):
        """z' = (1 - r) z + r f(z). gate=True: r = sigmoid(w . z + b), no gradient through r; gate='known': r = lost,
        the (B, n) indicators of the lost slots per modality (None: no slot is lost)."""
        if self.cfg['gate'] == 'known':
            if lost is None:
                return zh, zl
            return tuple(z + r.unsqueeze(-1) * (self.f[m](z) - z) for m, z, r in zip(MODALITIES, (zh, zl), lost))
        if self.gate is None:
            return zh, zl
        out = []
        for m, z in zip(MODALITIES, (zh, zl)):
            r = torch.sigmoid(self.gate[m](z)).detach()
            out.append(z + r * (self.f[m](z) - z))
        return out[0], out[1]

    def forward(self, h, l, on=None):
        """Class logits. on: the samples on which the pre-gate acts (draw_gate(); None = all, as at test time)."""
        lost = self.known_lost(h, l) if self.cfg['gate'] == 'known' else None
        return self.classify_tokens(*self.complete_tokens(*self.encode(h, l, on), lost))

    # ---- training branch
    def set_fill(self, mean_h, mean_l):
        """fill='mean': the per-band values that replace the masked inputs (the band means of the image)."""
        if self.cfg['fill'] != 'mean':
            raise RuntimeError("set_fill is for fill='mean'")
        for buf, v in ((self.fill_h, mean_h), (self.fill_l, mean_l)):
            buf.copy_(torch.as_tensor(v, dtype=buf.dtype).view_as(buf))

    def fill(self, h, l, mh, ml, bh=None):
        """The masked inputs: fill values at the masked positions (mh, ml: (B, n)) and, with bh ((B, ch), 1 = masked),
        in the masked HSI bands at every position."""
        w = self.window
        mh4, ml4 = mh.view(-1, 1, w, w), ml.view(-1, 1, w, w)
        h, l = h * (1 - mh4) + self.fill_h * mh4, l * (1 - ml4) + self.fill_l * ml4
        if bh is not None:
            b4 = bh[:, :, None, None]
            h = h * (1 - b4) + self.fill_h * b4
        return h, l

    def draw_masks(self, batch, g=None, device=None):
        c = self.cfg
        if c['mask_block'] == 1:                                          # pixel masks (drawn as before blocks existed)
            return make_masks(batch, self.n, c['mask_ratio'], c['mask_mode'], c['mask_split'], g, device)
        mh, ml = make_masks(batch, self.units, c['mask_ratio'], c['mask_mode'], c['mask_split'], g, device)
        idx = block_index(self.window, c['mask_block'], mh.device)      # block masks, spread to the pixels
        return mh[:, idx], ml[:, idx]

    def draw_bands(self, batch, g=None, device=None):
        """band_mask > 0: (B, ch), 1 = the band is masked in the whole patch; per sample a share U(0, band_mask) of
        the bands, drawn at random. None when band_mask = 0 (no random number is drawn then)."""
        b, ch = self.cfg['band_mask'], self.cfg['ch']
        if b <= 0:
            return None
        k = torch.round(uniform((batch,), g, device) * b * ch).long().to(device)
        return (rank(batch, ch, g, device) < k[:, None]).float()

    def masked_inputs(self, h, l, g=None):
        """A masked copy of a batch, with freshly drawn masks of the completion branch (mask_labeled=True)."""
        mh, ml = self.draw_masks(h.shape[0], g, h.device)
        return self.fill(h, l, mh, ml, self.draw_bands(h.shape[0], g, h.device))

    def pool_losses(self, h, l, g=None, masks=None, on=None):
        """Training-branch losses of a batch of pool patches. masks=(mh, ml) to give the masks instead of drawing;
        on: the samples whose masked view the pre-gate acts on (draw_gate(); None = all)."""
        if not self.branch:
            raise RuntimeError('this model has no training branch (decoder=None)')
        c = self.cfg
        mh, ml = masks if masks is not None else self.draw_masks(h.shape[0], g, h.device)
        bh = self.draw_bands(h.shape[0], g, h.device)                     # None unless band_mask > 0
        out = {'mh': mh, 'ml': ml}
        with self.masked_bn():
            if self.pgate is None:
                zh, zl = self.encode(*self.fill(h, l, mh, ml, bh), on)
            else:                                                         # keep the gate logits of the masked view
                hm, lm = self.fill(h, l, mh, ml, bh)
                th, tl = self.tokens(hm, lm)
                kb, (sh, sl) = self.key_bias(hm, lm, th, tl, on)
                zh, zl = self.fuse(th, tl, kb)
            uh, ul = (zh, zl) if self.dec is None else self.dec(zh, zl)
            xh, xl = h.flatten(2).transpose(1, 2), l.flatten(2).transpose(1, 2)
            rh, rl = (uh, ul) if c['rec_source'] == 'self' else (ul, uh)  # tokens read by the pixel heads g_H, g_L
            mse_h = masked_mean(((self.g['h'](rh) - xh) ** 2).mean(2), mh)
            mse_l = masked_mean(((self.g['l'](rl) - xl) ** 2).mean(2), ml)
            if self.s is None:
                out['rec'] = mse_h + mse_l
            elif self.s.numel() == 1:                                     # balanced: weights e^-s, e^s
                out['rec'] = torch.exp(-self.s[0]) * mse_h + torch.exp(self.s[0]) * mse_l
            else:                                                         # kendall
                out['rec'] = torch.exp(-self.s[0]) * mse_h + self.s[0] + torch.exp(-self.s[1]) * mse_l + self.s[1]
            out['mse_h'], out['mse_l'] = mse_h.detach(), mse_l.detach()
            if c['feat'] or c['cons']:
                with torch.no_grad():                                     # clean view: same weights, no gradient
                    if self.pgate is None:
                        ch_, cl_ = self.encode(h, l)
                    else:
                        tch, tcl = self.tokens(h, l)
                        ch_, cl_ = self.fuse(tch, tcl, self.key_bias(h, l, tch, tcl)[0])
                if self.pgate is not None:                                # detect the masked slots; clean view: none
                    bce = F.binary_cross_entropy_with_logits
                    sch, scl = self.lost_logits(tch, tcl)                 # gradient to the gate only
                    out['pre_gate'] = 0.5 * sum(bce(s, mk) + bce(sc, torch.zeros_like(mk))
                                                for s, sc, mk in ((sh, sch, mh), (sl, scl, ml)))
                if c['feat']:
                    ph, pl = (uh, ul) if self.f is None else (self.f['h'](uh), self.f['l'](ul))
                    out['feat'] = (masked_mean(((ph - ch_) ** 2).mean(2), mh)
                                   + masked_mean(((pl - cl_) ** 2).mean(2), ml))
                if self.gate is not None:                                 # detect the masked slots; clean view: none
                    bce = F.binary_cross_entropy_with_logits
                    out['gate'] = 0.5 * sum(bce(self.gate[m](z).squeeze(-1), mk)
                                            + bce(self.gate[m](zc).squeeze(-1), torch.zeros_like(mk))
                                            for m, z, zc, mk in (('h', zh, ch_, mh), ('l', zl, cl_, ml)))
                if c['cons']:
                    with torch.no_grad():
                        p = torch.softmax(self.classify_tokens(*self.complete_tokens(ch_, cl_)), 1)
                    logq = torch.log_softmax(self.classify_tokens(*self.complete_tokens(zh, zl, (mh, ml))), 1)
                    kl = (p * (torch.log(p.clamp_min(1e-8)) - logq)).sum(1)
                    keep = (p.max(1).values >= c['cons_tau']).float()
                    out['cons'] = (kl * keep).sum() / keep.sum().clamp_min(1.0)
        return out

    def loss(self, logits, y, pool=None, g=None):
        """Total loss of one step: labelled logits / labels, plus the training branch on pool=(h, l) if given."""
        ce = F.cross_entropy(logits, y)
        total, logged = ce, {'ce': ce}
        if pool is not None and self.branch:
            o = self.pool_losses(pool[0], pool[1], g)
            c = self.cfg
            total = total + c['lam'] * o['rec']
            if 'feat' in o:
                total = total + c['beta'] * o['feat']
            if 'cons' in o:
                total = total + c['gamma'] * o['cons']
            if 'gate' in o:
                total = total + o['gate']
            if 'pre_gate' in o:
                total = total + o['pre_gate']
            logged.update({k: o[k] for k in ('rec', 'mse_h', 'mse_l', 'feat', 'cons', 'gate', 'pre_gate') if k in o})
        values = torch.stack([v.detach().float() for v in logged.values()]).tolist()   # one GPU sync per step
        return total, dict(zip(logged, values))

    # ---- bookkeeping
    def is_test_time(self, name):
        gated = ('f.', 'gate.') if self.cfg['gate'] else ()             # gated completion keeps its heads
        return name.startswith(TEST_TIME_PREFIXES + gated + ('pgate.',))

    def count_parameters(self):
        test = sum(p.numel() for k, p in self.named_parameters() if self.is_test_time(k))
        return dict(test_time=test, training_only=sum(p.numel() for p in self.parameters()) - test)

    def strip_training_branch(self):
        """Delete everything used only in training (for deployment); the classifier's output does not change."""
        keep = ('f',) if self.cfg['gate'] else ()
        if 'known' in (self.cfg['pre_gate'], self.cfg['gate']):           # the fill values recognise the lost slots
            keep += ('fill_h', 'fill_l')
        for k in ('fill_h', 'fill_l', 'dec', 'g', 's', 'f'):
            if k not in keep and hasattr(self, k):
                delattr(self, k)
        self.dec, self.branch = None, False
        if 'f' not in keep:
            self.f = None
        return self
