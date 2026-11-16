"""Unit tests of symbionet/model.py on random data (CPU, a few seconds).

Run with `python tests/test_model.py` (prints a summary) or with pytest.
"""
import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import symbionet.model as M  # noqa: E402

W, N, CH, CL, K = 11, 121, 30, 2, 7


def data(B=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, CH, W, W, generator=g), torch.randn(B, CL, W, W, generator=g),
            torch.randint(0, K, (B,), generator=g))


def model(seed=0, **kw):
    torch.manual_seed(seed)
    return M.SymbioNet(CH, CL, K, window=W, **kw)


def test_shapes():
    for dec in M.DECODERS:
        net = model(decoder=dec)
        h, l, _ = data()
        zh, zl = net.encode(h, l)
        assert zh.shape == (4, N, 64) and zl.shape == (4, N, 64), dec
        assert net(h, l).shape == (4, K), dec


def test_masks():
    g = torch.Generator().manual_seed(1)
    B = 400
    mh, ml = M.make_masks(B, N, 0.5, 'independent', 'dirichlet', g, 'cpu')
    assert torch.all(mh.sum(1) + ml.sum(1) == 121) and (mh * ml).sum() > 0
    assert abs(float(mh.sum(1).mean()) - 60.5) < 4 and float(mh.sum(1).std()) > 25       # HSI share ~ U(0, 1)
    mh, ml = M.make_masks(B, N, 0.5, 'complementary', 'dirichlet', g, 'cpu')
    assert float((mh * ml).sum()) == 0 and torch.all(mh.sum(1) + ml.sum(1) == 121)        # M_l = 1 - M_h
    mh, ml = M.make_masks(B, N, 0.4, 'complementary', 'equal', g, 'cpu')
    V = int(round(0.4 * 2 * N))
    assert torch.all(mh.sum(1) == V // 2) and torch.all(ml.sum(1) == V - V // 2) and float((mh * ml).sum()) == 0
    mh, ml = M.make_masks(B, N, 0.5, 'same', 'dirichlet', g, 'cpu')
    assert torch.equal(mh, ml) and torch.all(mh.sum(1) == int(round(0.5 * N)))
    for mode, ratio in (('complementary', 0.6), ('nope', 0.5)):
        try:
            M.make_masks(2, N, ratio, mode, 'dirichlet', g, 'cpu')
            raise AssertionError('accepted %s %s' % (mode, ratio))
        except ValueError:
            pass
    try:
        model(mask_mode='complementary', mask_ratio=0.6)
        raise AssertionError('model accepted complementary masking at 0.6')
    except ValueError:
        pass



def test_block_masks():
    # mask_block=1 (the default) draws exactly the pixel masks of make_masks, so earlier runs are unchanged
    h, l, _ = data(B=6)
    net = model(decoder='identity', mask_mode='complementary')
    o = net.pool_losses(h, l, g=torch.Generator().manual_seed(3))
    ref = M.make_masks(6, N, 0.5, 'complementary', 'dirichlet', torch.Generator().manual_seed(3), 'cpu')
    assert torch.equal(o['mh'], ref[0]) and torch.equal(o['ml'], ref[1])
    for w in (9, 11):
        G = math.ceil(w / 3)
        idx = M.block_index(w, 3, 'cpu')
        assert idx.numel() == w * w and int(idx.max()) == G * G - 1 and int(idx[w * w // 2]) == (w // 2 // 3) * (G + 1)
        hh, ll = torch.randn(200, CH, w, w), torch.randn(200, CL, w, w)
        for mode in M.MASK_MODES:
            torch.manual_seed(0)
            net = M.SymbioNet(CH, CL, K, window=w, decoder='identity', mask_mode=mode, mask_block=3, feat=True)
            o = net.pool_losses(hh, ll, g=torch.Generator().manual_seed(4))
            mh, ml = o['mh'], o['ml']
            assert mh.shape == (200, w * w) and torch.isfinite(o['rec']) and torch.isfinite(o['feat'])
            for m in (mh, ml):                                            # one value inside every block
                for b in range(G * G):
                    col = m[:, idx == b]
                    assert torch.all(col == col[:, :1]), (w, mode, b)
            frac = float((mh.sum(1) + ml.sum(1)).mean()) / (2 * w * w)
            if mode == 'complementary':                                    # ratio 0.5: each pixel hidden in one
                assert float((mh * ml).sum()) == 0 and torch.all(mh + ml == 1), w
            elif mode == 'same':
                assert torch.equal(mh, ml) and abs(frac - 0.5) < 0.06, (w, frac)
            else:
                assert float((mh * ml).sum()) > 0 and abs(frac - 0.5) < 0.06, (w, frac)
    whole = M.SymbioNet(CH, CL, K, window=W, decoder='identity', mask_mode='complementary', mask_block=W)
    o = whole.pool_losses(*data(B=50)[:2], g=torch.Generator().manual_seed(5))
    assert torch.all((o['mh'].sum(1) == N) ^ (o['ml'].sum(1) == N))   # b = w: one whole modality per patch
    for bad in (dict(mask_block=0), dict(mask_block=W + 1), dict(mask_block=2.0),
                dict(mask_block=3, mask_mode='complementary', mask_ratio=0.6)):
        try:
            model(decoder='identity', **bad)
            raise AssertionError('accepted %s' % bad)
        except ValueError:
            pass

def test_fill():
    net = model()
    with torch.no_grad():
        net.fill_h.normal_()
        net.fill_l.normal_()
    h, l, _ = data()
    mh, ml = M.make_masks(4, N, 0.5, 'independent', 'dirichlet', torch.Generator().manual_seed(2), 'cpu')
    hf, lf = net.fill(h, l, mh, ml)
    mh4, ml4 = mh.view(4, 1, W, W).bool().expand_as(h), ml.view(4, 1, W, W).bool().expand_as(l)
    assert torch.equal(hf[~mh4], h[~mh4]) and torch.equal(lf[~ml4], l[~ml4])
    assert torch.allclose(hf, torch.where(mh4, net.fill_h.expand_as(h), h))
    assert torch.allclose(lf, torch.where(ml4, net.fill_l.expand_as(l), l))


def test_masked_mse_and_weightings():
    """With constant heads the reconstruction errors can be computed by hand; every weighting starts at the sum."""
    net = model(rec_weighting='none')
    with torch.no_grad():
        for m, cst in (('h', 0.3), ('l', -0.7)):
            net.g[m].weight.zero_()
            net.g[m].bias.fill_(cst)
    h, l, _ = data()
    mh, ml = M.make_masks(4, N, 0.5, 'independent', 'dirichlet', torch.Generator().manual_seed(3), 'cpu')
    out = net.pool_losses(h, l, masks=(mh, ml))
    eh = ((0.3 - h.flatten(2).transpose(1, 2)) ** 2).mean(2)
    el = ((-0.7 - l.flatten(2).transpose(1, 2)) ** 2).mean(2)
    exp_h, exp_l = (eh * mh).sum() / mh.sum(), (el * ml).sum() / ml.sum()
    assert torch.allclose(out['mse_h'], exp_h, atol=1e-6) and torch.allclose(out['mse_l'], exp_l, atol=1e-6)
    assert torch.allclose(out['rec'], exp_h + exp_l, atol=1e-6)
    for rw in ('kendall', 'balanced'):
        other = model(rec_weighting=rw)
        other.load_state_dict(net.state_dict(), strict=False)
        assert torch.allclose(other.pool_losses(h, l, masks=(mh, ml))['rec'], out['rec'], atol=1e-6), rw
    bal = model(rec_weighting='balanced')
    bal.load_state_dict(net.state_dict(), strict=False)
    with torch.no_grad():
        bal.s.fill_(0.5)
    r = bal.pool_losses(h, l, masks=(mh, ml))['rec']
    assert torch.allclose(r, math.exp(-0.5) * exp_h + math.exp(0.5) * exp_l, atol=1e-5)   # weights multiply to 1


def test_rec_source():
    """rec_source='cross': g_H reads the LiDAR tokens and g_L the HSI tokens at the same positions; 'self' (the
    default) reads the own tokens as before. L_feat is the same for both; both pass gradients to the heads."""
    h, l, y = data()
    mh, ml = M.make_masks(4, N, 0.5, 'complementary', 'dirichlet', torch.Generator().manual_seed(13), 'cpu')
    xh, xl = h.flatten(2).transpose(1, 2), l.flatten(2).transpose(1, 2)
    for dec in ('identity', 'mamba'):
        net = model(decoder=dec, rec_weighting='none', feat=True)
        cross = model(decoder=dec, rec_weighting='none', feat=True, rec_source='cross')
        cross.load_state_dict(net.state_dict())                    # same parameters, only the reading differs
        assert net.cfg['rec_source'] == 'self' and cross.cfg['rec_source'] == 'cross'
        with torch.no_grad():
            a = net.pool_losses(h, l, masks=(mh, ml))
            b = cross.pool_losses(h, l, masks=(mh, ml))
            with M.bn_frozen(net):                                  # the tokens pool_losses works on
                zh, zl = net.encode(*net.fill(h, l, mh, ml))
                uh, ul = (zh, zl) if net.dec is None else net.dec(zh, zl)

            def mse(head, u, x, m):
                return M.masked_mean(((head(u) - x) ** 2).mean(2), m)
            assert torch.allclose(a['mse_h'], mse(net.g['h'], uh, xh, mh), atol=1e-6), dec
            assert torch.allclose(a['mse_l'], mse(net.g['l'], ul, xl, ml), atol=1e-6), dec
            assert torch.allclose(b['mse_h'], mse(net.g['h'], ul, xh, mh), atol=1e-6), dec
            assert torch.allclose(b['mse_l'], mse(net.g['l'], uh, xl, ml), atol=1e-6), dec
            assert not torch.allclose(a['mse_h'], b['mse_h']), dec
            assert torch.equal(a['feat'], b['feat']), dec
        total, _ = cross.loss(cross(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(14))
        total.backward()
        assert all(float(cross.g[m].weight.grad.abs().sum()) > 0 for m in M.MODALITIES), dec
    try:
        model(decoder='identity', rec_source='both')
        raise AssertionError('accepted rec_source=both')
    except ValueError:
        pass


def test_bn_frozen():
    net = model(feat=True, cons=True)
    net.train()
    h, l, _ = data(8)
    bns = [b for b in net.modules() if isinstance(b, nn.BatchNorm2d)]
    before = copy.deepcopy([b.running_mean.clone() for b in bns])
    net.pool_losses(h, l, torch.Generator().manual_seed(4))
    assert all(torch.equal(a, b.running_mean) for a, b in zip(before, bns)), 'pool batch changed BN statistics'
    net(h, l)
    assert any(not torch.equal(a, b.running_mean) for a, b in zip(before, bns)), 'labelled batch left BN unchanged'
    assert all(b.momentum == 0.1 for b in bns)


def test_gradients():
    for dec in ('identity', 'transformer', 'mamba'):
        net = model(decoder=dec, feat=True, cons=True)
        h, l, y = data(6)
        ph, pl, _ = data(6, seed=9)
        total, parts = net.loss(net(h, l), y, pool=(ph, pl), g=torch.Generator().manual_seed(5))
        total.backward()
        need = ['stem.h.0.weight', 'stem.l.0.weight', 'pos', 'blocks.0.qkv.h.weight', 'blocks.1.qkv.l.weight',
                'head.weight', 'fill_h', 'fill_l', 'g.h.weight', 'g.l.weight', 's', 'f.h.weight', 'f.l.weight']
        if dec != 'identity':
            need.append(next(k for k, _ in net.named_parameters() if k.startswith('dec.')))
        params = dict(net.named_parameters())
        missing = [k for k in need if params[k].grad is None or float(params[k].grad.abs().sum()) == 0]
        assert not missing, (dec, missing)
        assert all(k in parts for k in ('ce', 'rec', 'mse_h', 'mse_l', 'feat', 'cons')) and math.isfinite(total.item())


def test_clean_target_carries_no_gradient():
    """The target of L_feat (the clean view) passes no gradient: replacing it by constants of the same value leaves the
    gradients unchanged, while letting gradient through it (negative control) changes them."""
    mh = torch.zeros(4, N)
    mh[:, ::2] = 1                                  # half of the positions masked: the encoder also sees real data
    masks = (mh, 1 - mh)
    h, l, _ = data(4)

    def grad(clean_view):                           # stem gradient of L_feat, the clean view given by clean_view()
        net = model(feat=True, rec_weighting='none')
        orig, calls = net.encode, {'n': 0}

        def enc(a, b, on=None):                     # first call: the masked view; second: the clean view
            calls['n'] += 1
            return orig(a, b, on) if calls['n'] == 1 else clean_view(net, orig, a, b)
        net.encode = enc
        net.pool_losses(h, l, masks=masks)['feat'].backward()
        assert calls['n'] == 2
        return net.stem['h'][0].weight.grad.clone()

    def as_constants(net, orig, a, b):
        with torch.no_grad(), M.bn_frozen(net):
            return tuple(z.clone() for z in orig(a, b))

    def with_gradient(net, orig, a, b):
        with torch.enable_grad():
            return orig(a, b)

    g0 = grad(lambda net, orig, a, b: orig(a, b))   # the model's own clean view
    assert g0.abs().max() > 0
    tol = 1e-6 * g0.abs().max().item()
    assert torch.allclose(g0, grad(as_constants), rtol=1e-5, atol=tol)
    assert not torch.allclose(g0, grad(with_gradient), rtol=1e-5, atol=tol)


def test_gelu_gradient_on_every_device():
    """The PyTorch in use computes GELU gradients correctly (on the CPU after the check in model.py, and on the GPU)."""
    for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
        assert M.gelu_gradient_ok(dev), dev


def test_gradients_are_reproducible():
    """The same model, data and masks give bit-identical gradients twice (no uninitialised memory anywhere)."""
    h, l, y = data(4)
    mh = torch.zeros(4, N)
    mh[:, ::2] = 1

    def grads():
        net = model(decoder='mamba', feat=True)
        total, _ = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(7))
        total.backward()
        return [p.grad.clone() for p in net.parameters() if p.grad is not None]

    a, b = grads(), grads()
    assert len(a) == len(b) and all(torch.equal(x, z) for x, z in zip(a, b))


def test_test_time_network_is_the_base():
    for dec in ('identity', 'transformer', 'mamba'):
        net = model(decoder=dec, feat=True)
        net.eval()
        h, l, _ = data()
        with torch.no_grad():
            a = net(h, l)
            base = model(decoder=None)
            base.load_state_dict({k: v for k, v in net.state_dict().items() if net.is_test_time(k)})
            base.eval()
            b = base(h, l)
            net.strip_training_branch()
            c = net(h, l)
        assert torch.equal(a, b) and torch.equal(a, c), dec
        assert all(net.is_test_time(k) for k, _ in net.named_parameters()), dec


def ref_scan(u, dt, A, Bm, Cm, D):
    B, L, c = u.shape
    y = torch.zeros_like(u)
    for b in range(B):
        for i in range(c):
            hs = torch.zeros(A.shape[1], dtype=u.dtype)
            for t in range(L):
                hs = torch.exp(dt[b, t, i] * A[i]) * hs + dt[b, t, i] * Bm[b, t] * u[b, t, i]
                y[b, t, i] = (hs * Cm[b, t]).sum() + D[i] * u[b, t, i]
    return y


def scan_inputs(B, L, c, S, dtype, seed=6):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(B, L, c, generator=g, dtype=dtype)
    dt = torch.rand(B, L, c, generator=g, dtype=dtype) * 0.5
    A = -torch.rand(c, S, generator=g, dtype=dtype) - 0.1
    Bm = torch.randn(B, L, S, generator=g, dtype=dtype)
    Cm = torch.randn(B, L, S, generator=g, dtype=dtype)
    D = torch.randn(c, generator=g, dtype=dtype)
    return [x.requires_grad_() for x in (u, dt, A, Bm, Cm, D)]


def test_selective_scan():
    with torch.no_grad():
        for L in (1, 2, 7):
            x = scan_inputs(2, L, 3, 2, torch.float64)
            assert torch.allclose(M.selective_scan(*x), ref_scan(*x)), L
        x = scan_inputs(2, 60, 4, 3, torch.float32)
        assert torch.allclose(M.selective_scan(*x).double(), ref_scan(*[t.double() for t in x]), atol=1e-5)
    for L in (1, 2, 7):                 # the hand-written backward pass against numerical gradients
        assert torch.autograd.gradcheck(M.selective_scan, tuple(scan_inputs(2, L, 3, 2, torch.float64))), L


def test_mamba_interleave_order():
    torch.manual_seed(7)
    dec = M.MambaDecoder(64, 128, 2)
    with torch.no_grad():                       # every block the identity
        for b in dec.blocks:
            for mix in (b.fwd, b.bwd):
                mix.out_proj.weight.zero_()
                mix.out_proj.bias.zero_()
            b.mlp[2].weight.zero_()
            b.mlp[2].bias.zero_()
        dec.mod.normal_()
    zh, zl = torch.randn(3, N, 64), torch.randn(3, N, 64)
    uh, ul = dec(zh, zl)
    assert torch.allclose(uh, dec.norm['h'](zh + dec.mod[0]), atol=1e-6)
    assert torch.allclose(ul, dec.norm['l'](zl + dec.mod[1]), atol=1e-6)


def test_sfb_two_inputs_two_outputs():
    torch.manual_seed(8)
    blk = M.SFB()
    th, tl = torch.randn(2, N, 64), torch.randn(2, N, 64)
    a_h, a_l = blk(th, tl)
    assert not torch.allclose(a_h, blk(th, tl + 1.0)[0]), 'HSI output ignores LiDAR'
    assert not torch.allclose(a_l, blk(th + 1.0, tl)[1]), 'LiDAR output ignores HSI'


def test_determinism():
    vals = []
    for _ in range(2):
        net = model(seed=3, decoder='mamba', feat=True, cons=True)
        h, l, y = data(4)
        ph, pl, _ = data(4, seed=11)
        total, _ = net.loss(net(h, l), y, pool=(ph, pl), g=torch.Generator().manual_seed(12))
        vals.append(total.item())
    assert vals[0] == vals[1], vals


def test_parameter_split():
    for dec in M.DECODERS:
        net = model(decoder=dec, feat=True)
        c = net.count_parameters()
        assert c['test_time'] + c['training_only'] == sum(p.numel() for p in net.parameters()), dec
    net = model(decoder='identity', feat=True, stem='spectral', rel_bias=True, readout='attn')
    c = net.count_parameters()
    assert c['test_time'] + c['training_only'] == sum(p.numel() for p in net.parameters())
    assert all(net.is_test_time(k) for k, _ in net.named_parameters() if k.startswith(('stem.', 'readout.'))
               or '.rel.' in k)


# ------------------------------------------------------------------------------- options of the deployed network
FINAL = dict(decoder='identity', rec_weighting='none', feat=True, lam=0.0, mask_mode='complementary')


def test_spectral_stem():
    torch.manual_seed(0)
    stem = M.SpectralStem(CH, 64).eval()
    h = torch.randn(2, CH, W, W)
    with torch.no_grad():
        y = stem(h)
        assert y.shape == (2, 64, W, W)
        c = W // 2
        for dist, changes in ((2, True), (3, False)):              # receptive field 5 x 5, as the conv stem
            h2 = h.clone()
            h2[:, :, c, c + dist] += 5.0
            assert (not torch.allclose(stem(h2)[:, :, c, c], y[:, :, c, c])) == changes, dist
    n_spec = sum(p.numel() for p in M.SpectralStem(144, 64).parameters())
    n_conv = sum(p.numel() for p in M.conv_stem(144, 64, 64).parameters())
    assert n_spec < n_conv / 2, (n_spec, n_conv)
    net = model(stem='spectral', **FINAL)
    net.train()
    hb, lb, yb = data(6)
    bn3 = [b for b in net.modules() if isinstance(b, nn.BatchNorm3d)]
    assert bn3
    before = [b.running_mean.clone() for b in bn3]
    total, _ = net.loss(net(hb, lb), yb, pool=data(6, seed=9)[:2], g=torch.Generator().manual_seed(5))
    total.backward()
    assert net.stem['h'].spec[0].weight.grad.abs().sum() > 0 and math.isfinite(total.item())
    net.pool_losses(hb, lb, torch.Generator().manual_seed(4))
    with torch.no_grad():
        net(hb, lb)                                                 # a labelled batch updates BN, as before
    assert any(not torch.equal(a, b.running_mean) for a, b in zip(before, bn3))


def test_relative_position_bias():
    idx = M.relative_index(3)
    assert idx.shape == (9, 9) and torch.all(idx.diagonal() == 12)  # offset (0, 0) of a 3 x 3 window
    assert idx[0, 8] == 24 and idx[8, 0] == 0                       # (+2, +2) and (-2, -2)
    oh = M.relative_onehot(3)
    table = torch.randn(4, 25)
    assert torch.allclose((table @ oh).view(4, 9, 9), table[:, idx])
    # zero tables: exactly the plain SFB
    plain = model(**FINAL)
    rel = model(rel_bias=True, **FINAL)
    missing, unexpected = rel.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all('.rel.' in k for k in missing)
    plain.eval()
    rel.eval()
    h, l, y = data()
    with torch.no_grad():
        assert torch.equal(plain(h, l), rel(h, l))
        rel.blocks[0].rel['hl'].data[:, M.relative_index(W)[0, 0]] = 3.0   # HSI reads the LiDAR token at its position
        assert not torch.allclose(plain(h, l), rel(h, l))
    rel.train()
    total, _ = rel.loss(rel(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(2))
    total.backward()
    for b in rel.blocks:
        assert all(b.rel[p].grad is not None and b.rel[p].grad.abs().sum() > 0 for p in M.REL_PAIRS)
    assert 'rel_onehot' not in rel.state_dict()                     # rebuilt from the window, not saved


def test_attention_readout():
    mean = model(**FINAL)
    attn = model(readout='attn', **FINAL)
    missing, unexpected = attn.load_state_dict(mean.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('readout.') for k in missing)
    attn.load_state_dict({**attn.state_dict(), **{k: torch.zeros_like(v) for k, v in attn.state_dict().items()
                                                  if k.startswith('readout.') and '.q.' in k}})
    mean.eval()
    attn.eval()
    h, l, y = data()
    with torch.no_grad():
        assert torch.allclose(mean(h, l), attn(h, l), atol=1e-6)    # W_q = 0: the mean
    attn.train()
    total, _ = attn.loss(attn(h, l), y)
    total.backward()
    assert attn.readout['h']['q'].weight.grad.abs().sum() > 0
    assert attn.readout['h']['k'].weight.grad.abs().sum() == 0      # no score gradient while q = 0 (first step)


def test_fusion_information_flow():
    """Which queries see the other modality: change one input, look at the other modality's encoder output."""
    flows = {'joint': (True, True), 'self': (False, False), 'to_hsi': (True, False), 'to_lidar': (False, True),
             'shared': (True, True), 'fixed': (True, True)}          # (HSI reads LiDAR, LiDAR reads HSI)
    assert set(flows) == set(M.FUSIONS)
    h, l, y = data()
    for f, (h_reads, l_reads) in flows.items():
        net = model(fusion=f, **FINAL).eval()
        with torch.no_grad():
            zh, zl = net.encode(h, l)
            zh2 = net.encode(h, l + 1.0)[0]                         # only the LiDAR input changes
            zl2 = net.encode(h + 1.0, l)[1]                         # only the HSI input changes
        assert torch.equal(zh, zh2) != h_reads, f
        assert torch.equal(zl, zl2) != l_reads, f
        net.train()
        total, _ = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(1))
        total.backward()
        assert math.isfinite(total.item()), f
        assert all(b.qkv[m].weight.grad.abs().sum() > 0 for b in net.blocks for m in b.qkv), f


def test_shared_fusion():
    joint, shared = model(fusion='joint', **FINAL), model(fusion='shared', **FINAL)
    keys = [k for k in shared.state_dict() if k.startswith('blocks.')]
    assert keys and all(k.split('.')[3] == 'h' for k in keys)       # one parameter set per block
    block = sum(p.numel() for p in joint.blocks.parameters())
    c_j, c_s = joint.count_parameters()['test_time'], shared.count_parameters()['test_time']
    assert c_j - c_s == block // 2 - 2 * 64, (c_j, c_s, block)     # half the blocks, plus the modality embedding
    assert shared.is_test_time('mod_emb') and 'mod_emb' not in joint.state_dict()
    torch.manual_seed(2)
    blk = M.SFB(fusion='shared')
    t = torch.randn(2, N, 64)
    a_h, a_l = blk(t, t)
    assert torch.equal(a_h, a_l)                                    # the same function for both modalities
    h, l, y = data()
    total, _ = shared.loss(shared(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(3))
    total.backward()
    assert shared.mod_emb.grad.abs().sum() > 0


def test_fixed_mix_and_alpha():
    """'fixed' = (intra + cross) / 2; the SFB's joint softmax = alpha intra + (1 - alpha) cross (the mixture)."""
    torch.manual_seed(5)
    B, H, d = 2, 4, 64
    dh = d // H
    blk = M.SFB(d, H, 128, fusion='fixed')
    th, tl = torch.randn(B, N, d), torch.randn(B, N, d)
    q, k, v = {}, {}, {}
    for m, t in (('h', th), ('l', tl)):
        q[m], k[m], v[m] = blk.qkv[m](blk.n1[m](t)).chunk(3, -1)
    intra, cross = M.attention(q['h'], k['h'], v['h'], H), M.attention(q['h'], k['l'], v['l'], H)
    x = th + blk.o['h'](0.5 * (intra + cross))
    assert torch.allclose(blk(th, tl)[0], x + blk.mlp['h'](blk.n2['h'](x)), atol=1e-6)

    def split(z):
        return z.reshape(B, N, H, dh).transpose(1, 2)               # (B, heads, n, dh)
    a = split(q['h']) @ split(k['h']).transpose(-2, -1) / math.sqrt(dh)
    b = split(q['h']) @ split(k['l']).transpose(-2, -1) / math.sqrt(dh)
    alpha = torch.sigmoid(torch.logsumexp(a, -1) - torch.logsumexp(b, -1)).unsqueeze(-1)
    mix = alpha * split(intra) + (1 - alpha) * split(cross)
    joint = M.attention(q['h'], torch.cat([k['h'], k['l']], 1), torch.cat([v['h'], v['l']], 1), H)
    assert torch.allclose(mix.transpose(1, 2).reshape(B, N, d), joint, atol=1e-5)


def test_options_reload_as_robustness():
    """Like robustness.py: rebuild from the config, load the full state, strip the training branch."""
    options = [dict(stem='spectral', rel_bias=True, readout='attn')] + [dict(fusion=f, rel_bias=True)
                                                                        for f in M.FUSIONS]
    options += [dict(fill='mean'), dict(fill='mean', feat_head='none'), dict(fill='mean', gate=True),
                dict(fill='mean', gate=True, cons=True, cons_tau=0.5), dict(fill='mean', mask_labeled=True),
                dict(fill='mean', pre_gate='learned'), dict(fill='mean', pre_gate='learned', rel_bias=True, cons=True),
                dict(fill='mean', pre_gate='learned', mask_labeled=True, mask_block=W, bn_masked='running'),
                dict(fill='mean', pre_gate='learned', gate=True, fusion='fixed'),
                dict(fill='mean', pre_gate='known'), dict(fill='mean', pre_gate='known', rel_bias=True, cons=True),
                dict(fill='mean', pre_gate='known', mask_labeled=True, mask_block=W, bn_masked='running'),
                dict(fill='mean', pre_gate='known', gate=True, fusion='fixed'),
                dict(fill='mean', gate='known'), dict(fill='mean', gate='known', cons=True, cons_tau=0.5),
                dict(fill='mean', gate='known', pre_gate='known', mask_labeled=True, mask_block=W,
                     bn_masked='running'),
                dict(fill='mean', pre_gate='known', pre_gate_p=0.5, mask_labeled=True),
                dict(fill='mean', pre_gate='learned', pre_gate_p=0.5),
                dict(fill='mean', mask_labeled=True, band_mask=0.7), dict(band_mask=0.5)]
    for opt in options:
        net = model(**opt, **FINAL)
        h, l, y = data()
        total, _ = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(3))
        total.backward()
        with torch.no_grad():
            for p in net.parameters():                              # move every parameter away from its start
                p.add_(0.01 * torch.randn_like(p))
        state = copy.deepcopy(net.state_dict())
        cfg = {k: v for k, v in net.cfg.items() if k not in ('ch', 'cl', 'n_classes', 'window')}
        torch.manual_seed(123)
        fresh = M.SymbioNet(CH, CL, K, window=W, **cfg)
        fresh.load_state_dict(state)                                # strict
        net.strip_training_branch().eval()
        fresh.strip_training_branch().eval()
        with torch.no_grad():
            assert torch.equal(net(h, l), fresh(h, l)), opt
        assert all(net.is_test_time(k) for k, _ in net.named_parameters()), opt


# ------------------------------------------------------------------------------------- options for lost data
def test_fill_mean():
    """fill='mean': fixed fill values (buffers), saved with the weights; everything else as the learned fill."""
    learned, net = model(**FINAL), model(fill='mean', **FINAL)
    assert 'fill_h' in dict(net.named_buffers()) and 'fill_h' not in dict(net.named_parameters())
    a, b = learned.state_dict(), net.state_dict()
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)   # same start (both fills are zero)
    mean_h, mean_l = torch.rand(CH), torch.rand(CL)
    net.set_fill(mean_h.numpy().astype('float64'), mean_l.numpy().astype('float64'))
    h, l, y = data()
    mh, ml = torch.zeros(4, N), torch.zeros(4, N)
    mh[:, 3], ml[:, 7] = 1, 1
    fh, fl = (x.flatten(2) for x in net.fill(h, l, mh, ml))
    assert torch.equal(fh[:, :, 3], mean_h.expand(4, CH)) and torch.equal(fh[:, :, 4], h.flatten(2)[:, :, 4])
    assert torch.equal(fl[:, :, 7], mean_l.expand(4, CL)) and torch.equal(fl[:, :, 3], l.flatten(2)[:, :, 3])
    total, _ = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(1))
    total.backward()
    assert math.isfinite(total.item()) and not net.fill_h.requires_grad
    assert torch.equal(net.state_dict()['fill_h'].flatten(), mean_h)
    try:
        learned.set_fill(mean_h, mean_l)
        raise AssertionError('set_fill must refuse a learned fill')
    except RuntimeError:
        pass
    net.strip_training_branch()
    assert not hasattr(net, 'fill_h') and all(net.is_test_time(k) for k, _ in net.named_parameters())


def test_feat_head_none():
    """feat_head='none': the encoder tokens of the masked view are compared with the clean tokens directly."""
    net = model(feat_head='none', **FINAL)
    assert net.f is None and not any(k.startswith('f.') for k in net.state_dict())
    assert net.count_parameters()['training_only'] == sum(p.numel() for k, p in net.named_parameters()
                                                          if k.startswith(('fill_', 'g.')))
    h, l, _ = data()
    o = net.pool_losses(h, l, torch.Generator().manual_seed(2))
    with torch.no_grad():
        ch_, cl_ = net.encode(h, l)
    zh, zl = net.encode(*net.fill(h, l, o['mh'], o['ml']))
    want = M.masked_mean(((zh - ch_) ** 2).mean(2), o['mh']) + M.masked_mean(((zl - cl_) ** 2).mean(2), o['ml'])
    assert torch.allclose(o['feat'], want, atol=1e-6)
    o['feat'].backward()
    assert net.blocks[0].qkv['h'].weight.grad.abs().sum() > 0


def test_gate():
    """gate=True: z' = (1 - r) z + r f(z); r learns from L_gate only; f and the gate stay at test time."""
    plain = model(fill='mean', **FINAL)
    net = model(fill='mean', gate=True, **FINAL)
    extra = 2 * (64 * 64 + 64) + 2 * (64 + 1)                       # f_h, f_l and the two gates
    assert net.count_parameters()['test_time'] == plain.count_parameters()['test_time'] + extra
    h, l, y = data()
    net.eval()
    with torch.no_grad():
        zh, zl = net.encode(h, l)
        for m in M.MODALITIES:
            net.gate[m].weight.zero_()
            net.gate[m].bias.fill_(-50.0)                           # r = 0: the plain classifier
        assert torch.equal(net(h, l), net.classify_tokens(zh, zl))
        for m in M.MODALITIES:
            net.gate[m].bias.fill_(50.0)                            # r = 1: the classifier reads f(z)
        assert torch.allclose(net(h, l), net.classify_tokens(net.f['h'](zh), net.f['l'](zl)), atol=1e-5)
    net = model(fill='mean', gate=True, **FINAL)
    net.train()
    with torch.no_grad():                                           # closed at the start: almost the plain network
        assert torch.allclose(torch.sigmoid(net.gate['h'](torch.randn(3, N, 64))), torch.tensor(0.018), atol=1e-3)
    gate_params = [p for m in M.MODALITIES for p in net.gate[m].parameters()]
    ce = torch.nn.functional.cross_entropy(net(h, l), y)
    assert all(g is None for g in torch.autograd.grad(ce, gate_params, allow_unused=True, retain_graph=True))
    o = net.pool_losses(h, l, torch.Generator().manual_seed(3))
    sp = torch.nn.functional.softplus                               # closed gate (logit -4) against the masks
    want = 0.5 * sum(mk.mean() * sp(torch.tensor(4.0)) + (2 - mk.mean()) * sp(torch.tensor(-4.0))
                     for mk in (o['mh'], o['ml']))
    assert torch.allclose(o['gate'], want, atol=1e-5), (o['gate'].item(), want.item())
    assert all(g.abs().sum() > 0 for g in torch.autograd.grad(o['gate'], gate_params, retain_graph=True))
    with torch.no_grad():
        net.gate['h'].weight.normal_(0, 0.1)                        # once w != 0 the encoder is trained too
    o = net.pool_losses(h, l, torch.Generator().manual_seed(3))
    assert torch.autograd.grad(o['gate'], net.stem['h'][0].weight)[0].abs().sum() > 0
    total, logged = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(3))
    assert 'gate' in logged and math.isfinite(total.item())
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    masks = (o['mh'], o['ml'])
    first = last = None
    for _ in range(40):                                             # the gate can learn these masks
        loss = net.pool_losses(h, l, masks=masks)['gate']
        opt.zero_grad()
        loss.backward()
        opt.step()
        first, last = (loss.item() if first is None else first), loss.item()
    assert last < 0.6 * first, (first, last)
    before = net.eval()(h, l)
    net.strip_training_branch()
    assert net.f is not None and net.gate is not None and not hasattr(net, 'g') and not hasattr(net, 'fill_h')
    assert torch.equal(before, net(h, l))
    for bad in (dict(decoder='mamba'), dict(feat=False), dict(feat_head='none')):
        try:
            model(**{**FINAL, 'gate': True, **bad})
            raise AssertionError('gate must refuse %s' % bad)
        except ValueError:
            pass


def test_mask_labeled():
    """A masked copy of a labeled batch: the masks of the completion branch, also without the branch."""
    net = model(decoder=None, fill='mean', mask_labeled=True, mask_mode='complementary')
    assert not net.branch and net.count_parameters()['training_only'] == 0
    net.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    h, l, _ = data()
    mh_, ml_ = net.masked_inputs(h, l, torch.Generator().manual_seed(0))
    hit_h, hit_l = (mh_ == 7.0).all(1).flatten(1), (ml_ == 9.0).all(1).flatten(1)
    assert torch.all(hit_h.sum(1) + hit_l.sum(1) == N) and not (hit_h & hit_l).any()
    assert torch.equal(mh_[~hit_h.view(4, 1, W, W).expand_as(h)], h[~hit_h.view(4, 1, W, W).expand_as(h)])
    again = net.masked_inputs(h, l, torch.Generator().manual_seed(0))
    assert torch.equal(mh_, again[0]) and torch.equal(ml_, again[1])
    assert net(mh_, ml_).shape == (4, K)
    try:
        model(decoder=None, mask_labeled=True)
        raise AssertionError('mask_labeled without the branch needs fill=mean')
    except ValueError:
        pass


def test_gate_known():
    """gate='known': at the slots whose input equals the fill value the classifier reads f(z), elsewhere z."""
    plain = model(fill='mean', **FINAL)
    net = model(fill='mean', gate='known', **FINAL)
    assert net.gate is None and list(net.state_dict()) == list(plain.state_dict())
    extra = 2 * (64 * 64 + 64)                                      # f_h, f_l stay at test time
    assert net.count_parameters()['test_time'] == plain.count_parameters()['test_time'] + extra
    for n_ in (plain, net):
        n_.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    plain.load_state_dict(net.state_dict())
    h, l, y = data()
    net.eval(), plain.eval()
    with torch.no_grad():
        assert torch.equal(net(h, l), plain(h, l))                  # intact data: exactly the plain network
        mh, ml = net.draw_masks(4, torch.Generator().manual_seed(2))
        hm, lm = net.fill(h, l, mh, ml)
        zh, zl = net.encode(hm, lm)
        want = net.classify_tokens(torch.where(mh.bool().unsqueeze(-1), net.f['h'](zh), zh),
                                   torch.where(ml.bool().unsqueeze(-1), net.f['l'](zl), zl))
        assert torch.allclose(net(hm, lm), want, atol=1e-5)
        assert not torch.allclose(net(hm, lm), plain(hm, lm), atol=1e-3)
    net.train()
    ce = torch.nn.functional.cross_entropy(net(hm, lm), y)          # the head f is trained by the classification too
    assert torch.autograd.grad(ce, net.f['h'].weight)[0].abs().sum() > 0
    total, logged = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(3))
    assert 'gate' not in logged and math.isfinite(total.item())
    cons = model(fill='mean', gate='known', cons=True, **FINAL)
    assert math.isfinite(cons.pool_losses(h, l, torch.Generator().manual_seed(3))['cons'].item())
    before = net.eval()(hm, lm)
    net.strip_training_branch()
    assert net.f is not None and hasattr(net, 'fill_h') and not hasattr(net, 'g')
    assert torch.equal(before, net(hm, lm))
    for bad in (dict(fill='learned'), dict(feat_head='none'), dict(gate='sometimes')):
        try:
            model(**{**FINAL, 'fill': 'mean', 'gate': 'known', **bad})
            raise AssertionError('gate must refuse %s' % bad)
        except ValueError:
            pass


def test_pre_gate_known():
    """pre_gate='known': slots whose input equals the fill value are not read by the other tokens."""
    plain = model(fill='mean', **FINAL)
    net = model(fill='mean', pre_gate='known', **FINAL)
    assert net.count_parameters() == plain.count_parameters() and net.pgate is None
    assert list(net.state_dict()) == list(plain.state_dict())
    for n_ in (plain, net):
        n_.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    plain.load_state_dict(net.state_dict())
    h, l, y = data()
    net.eval(), plain.eval()
    with torch.no_grad():
        assert torch.equal(net(h, l), plain(h, l))                  # intact data: exactly the plain network
        g = torch.Generator().manual_seed(2)
        mh, ml = net.draw_masks(4, g)
        hm, lm = net.fill(h, l, mh, ml)
        kh, kl = net.known_lost(hm, lm)
        assert torch.equal(kh, mh) and torch.equal(kl, ml)          # the masked slots, recognised from the input
        part = hm.clone()
        part[:, 0] = h[:, 0]                                        # one band still intact: not lost
        assert net.known_lost(part, lm)[0].sum() == 0
        assert not torch.allclose(net(hm, lm), plain(hm, lm), atol=1e-3)
        # the whole LiDAR lost: the HSI stream is the one of a network whose HSI tokens read HSI only
        lost_l = torch.full_like(l, 9.0)
        one_way = model(fill='mean', fusion='to_lidar', **FINAL).eval()
        one_way.load_state_dict(net.state_dict())
        assert torch.allclose(net.encode(h, lost_l)[0], one_way.encode(h, lost_l)[0], atol=1e-5)
        # both modalities lost everywhere: every key has the same bias, the plain network
        lost_h = torch.full_like(h, 7.0)
        assert torch.allclose(net(lost_h, lost_l), plain(lost_h, lost_l), atol=1e-5)
    net.train()
    o = net.pool_losses(h, l, torch.Generator().manual_seed(3))
    assert 'pre_gate' not in o and math.isfinite(o['feat'].item())
    assert torch.autograd.grad(o['feat'], net.stem['h'][0].weight)[0].abs().sum() > 0
    before = net.eval()(hm, lm)
    net.strip_training_branch()
    assert hasattr(net, 'fill_h') and hasattr(net, 'fill_l') and not hasattr(net, 'g') and net.f is None
    assert torch.equal(before, net(hm, lm))
    base = model(decoder=None, fill='mean', pre_gate='known', mask_labeled=True)    # also without the branch
    assert base(h, l).shape == (4, K)
    for bad in (dict(fill='learned'), dict(pre_gate='sometimes')):
        try:
            model(**{**FINAL, 'fill': 'mean', 'pre_gate': 'known', **bad})
            raise AssertionError('pre_gate must refuse %s' % bad)
        except ValueError:
            pass


def test_pre_gate_p():
    """pre_gate_p < 1: in training the pre-gate acts on a random part of the masked views; at test time on all."""
    kw = dict(fill='mean', pre_gate='known', **FINAL)
    full, half, plain = model(**kw), model(pre_gate_p=0.5, **kw), model(fill='mean', **FINAL)
    assert list(half.state_dict()) == list(full.state_dict()) == list(plain.state_dict())
    assert half.count_parameters() == full.count_parameters() and full.cfg['pre_gate_p'] == 1.0
    for n_ in (full, half, plain):
        n_.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    half.load_state_dict(full.state_dict())
    plain.load_state_dict(full.state_dict())
    # p = 1 or no pre-gate: nothing is drawn (the generator is untouched): the earlier training, bit for bit
    g = torch.Generator().manual_seed(5)
    state = g.get_state()
    assert full.draw_gate(8, g) is None and plain.draw_gate(8, g) is None and torch.equal(g.get_state(), state)
    on = half.draw_gate(4000, g)
    assert on.shape == (4000,) and set(on.unique().tolist()) == {0.0, 1.0} and abs(float(on.mean()) - 0.5) < 0.03
    assert torch.equal(on, half.draw_gate(4000, torch.Generator().manual_seed(5)))            # reproducible
    h, l, y = data()
    hm, lm = full.fill(h, l, *full.draw_masks(4, torch.Generator().manual_seed(2)))
    for n_ in (full, half, plain):
        n_.eval()
    with torch.no_grad():
        gated, ungated = full(hm, lm), plain(hm, lm)
        assert not torch.allclose(gated, ungated, atol=1e-3)
        assert torch.equal(half(hm, lm), gated)                     # test time: the gate acts on every sample
        assert torch.equal(half(hm, lm, on=torch.ones(4)), gated)
        assert torch.allclose(half(hm, lm, on=torch.zeros(4)), ungated, atol=1e-6)     # off: lost slots read as usual
        mixed = half(hm, lm, on=torch.tensor([1.0, 0.0, 0.0, 1.0]))
        assert torch.allclose(mixed[[0, 3]], gated[[0, 3]], atol=1e-6)
        assert torch.allclose(mixed[[1, 2]], ungated[[1, 2]], atol=1e-6)
        assert torch.equal(half(h, l, on=torch.zeros(4)), half(h, l))                  # intact data: nothing to gate
    # training branch: `on` reaches the masked view of the pool patches
    for n_ in (full, half, plain):
        n_.train()
    masks = full.draw_masks(4, torch.Generator().manual_seed(3))
    f_on = half.pool_losses(h, l, masks=masks, on=torch.ones(4))['feat']
    f_off = half.pool_losses(h, l, masks=masks, on=torch.zeros(4))['feat']
    assert torch.allclose(f_on, full.pool_losses(h, l, masks=masks)['feat'], atol=1e-6)
    assert torch.allclose(f_off, plain.pool_losses(h, l, masks=masks)['feat'], atol=1e-6)
    assert not torch.allclose(f_on, f_off, atol=1e-4)
    assert torch.autograd.grad(f_off, half.stem['h'][0].weight)[0].abs().sum() > 0
    # the learned gate: a sample that is switched off gets no key bias
    lg = model(fill='mean', pre_gate='learned', pre_gate_p=0.5, **FINAL)
    with torch.no_grad():
        for m in M.MODALITIES:
            lg.pgate[m].bias.fill_(0.5)                              # an open gate, so that the bias matters
    th, tl = lg.tokens(h, l)
    kb = lg.key_bias(h, l, th, tl, torch.tensor([1.0, 0.0, 1.0, 0.0]))[0]
    assert all(float(kb[m][[1, 3]].abs().max()) == 0 and float(kb[m][[0, 2]].abs().min()) > 0 for m in M.MODALITIES)
    # the weights of such a model tested without the gate (robustness.py --set model.pre_gate=false): the plain network
    off = model(**{**kw, 'pre_gate': False, 'pre_gate_p': 0.5}).eval()
    off.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    off.load_state_dict(half.state_dict())
    with torch.no_grad():
        assert off.draw_gate(8) is None and torch.equal(off(hm, lm), ungated)
    for bad in (dict(pre_gate_p=0.0), dict(pre_gate_p=1.5), dict(pre_gate_p=True), dict(pre_gate_p='half')):
        try:
            model(**{**kw, **bad})
            raise AssertionError('pre_gate_p must refuse %s' % bad)
        except ValueError:
            pass


def test_band_mask():
    """band_mask > 0: the masked views also lose a random share of the HSI bands in the whole patch."""
    kw = dict(fill='mean', mask_labeled=True, **FINAL)
    plain, net = model(**kw), model(band_mask=0.7, **kw)
    assert list(net.state_dict()) == list(plain.state_dict()) and net.count_parameters() == plain.count_parameters()
    for n_ in (plain, net):
        n_.set_fill(torch.full((CH,), 7.0), torch.full((CL,), 9.0))
    net.load_state_dict(plain.state_dict())
    g = torch.Generator().manual_seed(5)
    state = g.get_state()
    assert plain.draw_bands(8, g) is None and torch.equal(g.get_state(), state)       # 0: nothing is drawn
    bh = net.draw_bands(2000, g)
    share = bh.mean(1)
    assert bh.shape == (2000, CH) and set(bh.unique().tolist()) == {0.0, 1.0}
    assert float(share.max()) <= 0.7 + 0.5 / CH and float(share.min()) == 0 and abs(float(share.mean()) - 0.35) < 0.03
    assert torch.equal(bh, net.draw_bands(2000, torch.Generator().manual_seed(5)))    # reproducible
    h, l, y = data()
    mh, ml = net.draw_masks(4, torch.Generator().manual_seed(2))
    b = torch.zeros(4, CH)
    b[:, ::3] = 1                                                                     # every third band masked
    hm, lm = net.fill(h, l, mh, ml, b)
    h0, l0 = net.fill(h, l, mh, ml)
    assert torch.equal(lm, l0) and torch.equal(net.fill(h, l, mh, ml, None)[0], h0)   # the LiDAR is not touched
    assert torch.all(hm[:, ::3] == 7.0)                                               # masked bands: the fill value
    keep = torch.ones(CH, dtype=torch.bool)
    keep[::3] = False
    assert torch.equal(hm[:, keep], h0[:, keep])                                      # the other bands: as before
    assert net.known_lost(hm, lm)[0].sum() == mh.sum()                                # lost slots: the masked positions
    # the masked copy and the pool losses use the band masks; the deployed network does not know about them
    gm = torch.Generator().manual_seed(9)
    a_h, _ = net.masked_inputs(h, l, gm)
    gm = torch.Generator().manual_seed(9)
    p_h, _ = plain.masked_inputs(h, l, gm)
    assert not torch.equal(a_h, p_h) and float((a_h == 7.0).float().mean()) > float((p_h == 7.0).float().mean())
    net.train()
    o = net.pool_losses(h, l, torch.Generator().manual_seed(3))
    assert math.isfinite(o['feat'].item())
    assert torch.autograd.grad(o['feat'], net.stem['h'][0].weight)[0].abs().sum() > 0
    net.eval(), plain.eval()
    with torch.no_grad():
        assert torch.equal(net(h, l), plain(h, l)) and torch.equal(net(hm, lm), plain(hm, lm))
    for bad in (dict(band_mask=-0.1), dict(band_mask=1.5), dict(band_mask=True), dict(band_mask='half')):
        try:
            model(**{**kw, **bad})
            raise AssertionError('band_mask must refuse %s' % bad)
        except ValueError:
            pass


def test_pre_gate():
    """pre_gate='learned': r from the stem tokens; every SFB multiplies the attention weight of a key by 1 - r."""
    plain = model(fill='mean', **FINAL)
    net = model(fill='mean', pre_gate='learned', **FINAL)
    assert net.count_parameters()['test_time'] == plain.count_parameters()['test_time'] + 2 * (64 + 1)
    assert net.count_parameters()['training_only'] == plain.count_parameters()['training_only']
    h, l, y = data()
    # closed gate (every key the same bias): exactly the plain network with the same weights
    plain.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith('pgate.')})
    net.eval(), plain.eval()
    with torch.no_grad():
        assert torch.allclose(net(h, l), plain(h, l), atol=1e-5)
        th, tl = net.tokens(h, l)
        sh, sl = net.lost_logits(th, tl)
        assert sh.shape == sl.shape == (4, N) and torch.allclose(torch.sigmoid(sh), torch.tensor(0.018), atol=1e-3)
        # LiDAR declared lost everywhere: the HSI stream is the one of a network whose HSI tokens read HSI only
        net.pgate['l'].bias.fill_(30.0)
        one_way = model(fill='mean', fusion='to_lidar', **FINAL).eval()
        one_way.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith('pgate.')})
        assert torch.allclose(net.encode(h, l)[0], one_way.encode(h, l)[0], atol=1e-4)
        assert not torch.allclose(net.encode(h, l)[0], plain.encode(h, l)[0], atol=1e-3)
        # ... and the LiDAR input no longer reaches the HSI stream
        assert torch.allclose(net.encode(h, l)[0], net.encode(h, torch.randn_like(l))[0], atol=1e-4)
        # one lost key: its attention weight is multiplied by 1 - r (two keys with equal scores)
        q, k2, v2 = torch.zeros(1, 1, 4), torch.zeros(1, 2, 4), torch.tensor([[[1.0] * 4, [3.0] * 4]])
        r = torch.tensor(0.75)
        got = M.attention(q, k2, v2, 2, None, torch.stack([torch.tensor(0.0), torch.log(1 - r)])[None])
        assert torch.allclose(got, torch.full((1, 1, 4), (1 * 1.0 + 3 * 0.25) / 1.25))
    net = model(fill='mean', pre_gate='learned', **FINAL)
    net.train()
    gate_params = [p for m in M.MODALITIES for p in net.pgate[m].parameters()]
    with torch.no_grad():
        net.pgate['h'].weight.normal_(0, 0.1)                       # r depends on the tokens
    ce = torch.nn.functional.cross_entropy(net(h, l), y)            # the classification loss does not train the gate
    assert all(g is None for g in torch.autograd.grad(ce, gate_params, allow_unused=True, retain_graph=True))
    o = net.pool_losses(h, l, torch.Generator().manual_seed(3))
    assert all(g is None for g in torch.autograd.grad(o['feat'], gate_params, allow_unused=True, retain_graph=True))
    assert all(g.abs().sum() > 0 for g in torch.autograd.grad(o['pre_gate'], gate_params, retain_graph=True))
    assert torch.autograd.grad(o['pre_gate'], net.stem['h'][0].weight, retain_graph=True)[0].abs().sum() > 0
    net2 = model(fill='mean', pre_gate='learned', **FINAL)          # closed gate (logit -4) against the masks
    o2 = net2.pool_losses(h, l, torch.Generator().manual_seed(3))
    sp = torch.nn.functional.softplus
    want = 0.5 * sum(mk.mean() * sp(torch.tensor(4.0)) + (2 - mk.mean()) * sp(torch.tensor(-4.0))
                     for mk in (o2['mh'], o2['ml']))
    assert torch.allclose(o2['pre_gate'], want, atol=1e-5), (o2['pre_gate'].item(), want.item())
    total, logged = net.loss(net(h, l), y, pool=(h, l), g=torch.Generator().manual_seed(3))
    assert 'pre_gate' in logged and math.isfinite(total.item())
    # whole-modality masks: the gate learns to tell a lost modality (band means) from intact data
    net = model(fill='mean', pre_gate='learned', mask_block=W, **FINAL)
    net.set_fill(torch.zeros(CH), torch.zeros(CL))
    net.train()
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    hb, lb, _ = data(16, seed=5)
    masks = net.draw_masks(16, torch.Generator().manual_seed(1))
    assert all(bool(((m.sum(1) == 0) | (m.sum(1) == N)).all()) for m in masks)
    first = last = None
    for _ in range(80):
        loss = net.pool_losses(hb, lb, masks=masks)['pre_gate']
        opt.zero_grad()
        loss.backward()
        opt.step()
        first, last = (loss.item() if first is None else first), loss.item()
    assert last < 0.5 * first, (first, last)
    before = net.eval()(h, l)
    net.strip_training_branch()
    assert net.pgate is not None and not hasattr(net, 'g')
    assert torch.equal(before, net(h, l))
    for bad in (dict(decoder=None, fill='mean'), dict(feat=False)):
        try:
            model(**{**FINAL, 'pre_gate': 'learned', **bad})
            raise AssertionError('pre_gate must refuse %s' % bad)
        except ValueError:
            pass


def test_bn_masked_running():
    """bn_masked='running': the masked views use the running BatchNorm statistics (as at test time), unchanged."""
    net = model(fill='mean', bn_masked='running', mask_labeled=True, **FINAL)
    net.train()
    h, l, y = data(8)
    for _ in range(3):
        net(h, l)                                                   # clean batches move the running statistics
    bns = [b for b in net.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    before = [(b.running_mean.clone(), b.running_var.clone(), b.num_batches_tracked.clone()) for b in bns]
    with net.masked_bn():
        assert not any(b.training for b in bns)
        inside = net(*net.masked_inputs(h, l, torch.Generator().manual_seed(0)))
    assert all(b.training for b in bns) and net.training
    assert all(torch.equal(a, b.running_mean) and torch.equal(v, b.running_var)
               and torch.equal(c, b.num_batches_tracked) for (a, v, c), b in zip(before, bns))
    net.eval()                                                      # the same numbers as the test-time network
    with torch.no_grad():
        assert torch.allclose(inside, net(*net.masked_inputs(h, l, torch.Generator().manual_seed(0))), atol=1e-5)
    net.train()
    o = net.pool_losses(h, l, torch.Generator().manual_seed(4))
    assert all(torch.equal(a, b.running_mean) for (a, _, _), b in zip(before, bns)) and all(b.training for b in bns)
    assert torch.autograd.grad(o['feat'], net.stem['h'][0].weight)[0].abs().sum() > 0
    batch = model(fill='mean', mask_labeled=True, **FINAL)          # 'batch' (default): the statistics of the batch
    batch.load_state_dict(net.state_dict())
    batch.train()
    with batch.masked_bn():
        assert all(b.training for b in batch.modules() if isinstance(b, nn.modules.batchnorm._BatchNorm))
        other = batch(*batch.masked_inputs(h, l, torch.Generator().manual_seed(0)))
    assert not torch.allclose(inside, other, atol=1e-3)
    try:
        model(bn_masked='sometimes')
        raise AssertionError('unknown bn_masked')
    except ValueError:
        pass


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print('PASS', name)
        except Exception as e:                  # noqa: BLE001 - report every failure, keep going
            failed.append(name)
            print('FAIL', name, '-', type(e).__name__, e)
    print('=' * 60)
    print('test_model: %d/%d passed (torch %s)' % (len(tests) - len(failed), len(tests), torch.__version__))
    for dec in M.DECODERS:
        c = model(decoder=dec, feat=True).count_parameters()
        print('  parameters, decoder=%s: test time %d, training only %d' % (dec, c['test_time'], c['training_only']))
    for opt in [dict(stem='spectral'), dict(rel_bias=True), dict(readout='attn')] + [dict(fusion=f)
                                                                                    for f in M.FUSIONS[1:]] + [
            dict(fill='mean'), dict(fill='mean', feat_head='none'), dict(fill='mean', gate=True),
            dict(fill='mean', pre_gate='learned'), dict(fill='mean', pre_gate='known'),
            dict(fill='mean', pre_gate='known', pre_gate_p=0.5), dict(fill='mean', band_mask=0.7)]:
        c = model(feat=True, decoder='identity', **opt).count_parameters()
        print('  parameters, %s: test time %d, training only %d' % (opt, c['test_time'], c['training_only']))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
