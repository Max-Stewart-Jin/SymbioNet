"""Unit tests of completion_check.py with stand-in models whose completion error is known in closed form. Run with
`python tests/test_completion_check.py` or with pytest."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import completion_check as CC  # noqa: E402
from symbionet.model import make_masks  # noqa: E402

W, N = 5, 25


class Copy(torch.nn.Module):
    """Every token holds the HSI value and the LiDAR value of its position, so a masked token is "completed" exactly
    as far as the other modality is visible there. Inputs: one band each; the fill value is 0."""

    def __init__(self, leak=0.0):
        super().__init__()
        self.f = {'h': torch.nn.Identity(), 'l': torch.nn.Identity()}
        self.leak = leak                                 # share of the patch mean mixed into every token (context)

    def draw_masks(self, batch, g=None, device=None):
        return make_masks(batch, N, 0.5, 'complementary', 'dirichlet', g, device)

    def fill(self, h, l, mh, ml, bh=None):
        return h * (1 - mh.view(-1, 1, W, W)), l * (1 - ml.view(-1, 1, W, W))

    def encode(self, h, l, on=None):
        z = torch.stack([h.flatten(1), l.flatten(1)], 2)                     # (B, n, 2)
        z = (1 - self.leak) * z + self.leak * z.mean(1, keepdim=True)
        return z, z.clone()


class Own(Copy):
    """A model whose tokens hold their own modality only."""

    def encode(self, h, l, on=None):
        return h.flatten(1)[..., None].repeat(1, 1, 2), l.flatten(1)[..., None].repeat(1, 1, 2)


class Normed(Copy):
    """Copy with a BatchNorm layer on each input (running statistics far from those of the data)."""

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.ModuleDict({m: torch.nn.BatchNorm2d(1) for m in 'hl'})
        for b in self.bn.values():
            b.running_mean.fill_(5.0)
            b.running_var.fill_(9.0)
        self.eval()

    def encode(self, h, l, on=None):
        return super().encode(self.bn['h'](h), self.bn['l'](l))


def sampler(idx):
    g = torch.Generator().manual_seed(int(idx[0]))
    return 1 + torch.rand(len(idx), 1, W, W, generator=g), 2 + torch.rand(len(idx), 1, W, W, generator=g)


def rates(acc, m):
    return {k: acc[m]['position'][k][0] / acc[m]['position'][k][2] for k in CC.GROUPS}


def states(acc, m, b='all'):
    return {s: acc[m]['modality'][s][b][0] / acc[m]['modality'][s][b][1] for s in CC.STATES}


def test_groups_and_ratio():
    idx = torch.arange(400)
    acc = CC.measure(Copy(), sampler, idx, 64)
    for m in 'hl':
        n = {k: acc[m]['position'][k][2] for k in CC.GROUPS}
        assert abs(n['visible'] + n['hidden'] - n['train']) < 1e-6, n      # B splits the masked positions of A
        assert 0.45 < n['hidden'] / n['train'] < 0.55, n                    # about one half is hidden as well
    # complementary masks at ratio 0.5: every position is masked in exactly one modality
    assert abs(acc['h']['position']['train'][2] + acc['l']['position']['train'][2] - 400 * N) < 1e-6
    # HSI token = (h, l): masked h -> (0, l). Its error: mean(h^2, 0) if the LiDAR is visible there, mean(h^2, l^2) if
    # it is hidden as well; so hidden > visible, and A (as trained) equals the visible group
    e = rates(acc, 'h')
    assert abs(e['train'] - e['visible']) < 0.02 * e['train'], e
    assert e['hidden'] > 2.5 * e['visible'], e                               # l in [2, 3], h in [1, 2]
    e = rates(acc, 'l')
    assert abs(e['train'] - e['visible']) < 0.02 * e['train'], e
    assert 1.1 < e['hidden'] / e['visible'] < 1.6, e                         # the hidden h adds less than the lost l


def test_whole_modality():
    acc = CC.measure(Copy(), sampler, torch.arange(600), 64)
    # h ~ U(1, 2): E h^2 = 7/3; l ~ U(2, 3): E l^2 = 19/3. An HSI token (h, l) with h masked: error h^2 / 2 while the
    # LiDAR is there (shown, or as trained: complementary masks leave it visible at every masked position), and
    # (h^2 + l^2) / 2 with the LiDAR hidden everywhere
    e = states(acc, 'h')
    assert abs(e['shown'] - 7 / 6) < 0.03 and abs(e['train'] - e['shown']) < 1e-6, e
    assert abs(e['hidden'] / e['train'] - (7 / 3 + 19 / 3) / (7 / 3)) < 0.05, e
    e = states(acc, 'l')
    assert abs(e['shown'] - 19 / 6) < 0.05 and abs(e['train'] - e['shown']) < 1e-6, e
    assert abs(e['hidden'] / e['train'] - (7 / 3 + 19 / 3) / (19 / 3)) < 0.03, e
    for m in 'hl':                                                           # the three shares split all the positions
        n = {b: acc[m]['modality']['train'][b][1] for b in CC.BINS}
        assert abs(n['low'] + n['mid'] + n['high'] - n['all']) < 1e-6 and min(n.values()) > 0, n
        assert n['low'] < n['mid'] < n['high'], n                           # more positions where more is masked
        assert abs(acc[m]['modality']['train']['all'][0] - acc[m]['position']['train'][0]) < 1e-6
    row = CC.summarise(acc)
    assert abs(row['h']['modality']['ratio']['all'] - 26 / 7) < 0.05
    assert row['h']['modality']['positions']['all'] == int(acc['h']['position']['train'][2])


def test_shares_are_exact():
    """The bin of a patch follows from its number of masked positions (3 k against n and 2 n), also at the edges."""
    class Fixed(Copy):
        def draw_masks(self, batch, g=None, device=None):
            k = torch.tensor([0, 8, 9, 16, 17, 25])[:batch]                  # of n = 25: 9 > 25/3, 17 > 50/3
            mh = (torch.arange(N)[None] < k[:, None]).float()
            return mh, 1 - mh
    acc = CC.measure(Fixed(), sampler, torch.arange(6), 6)
    n = {b: acc['h']['modality']['train'][b][1] for b in CC.BINS}
    assert (n['low'], n['mid'], n['high'], n['all']) == (8, 9 + 16, 17 + 25, 75), n
    n = {b: acc['l']['modality']['train'][b][1] for b in CC.BINS}           # LiDAR: 25, 17, 16, 9, 8, 0 masked
    assert (n['low'], n['mid'], n['high'], n['all']) == (8, 16 + 9, 25 + 17, 75), n


def test_a_model_that_ignores_the_other_modality():
    acc = CC.measure(Own(), sampler, torch.arange(400), 64)
    for m in 'hl':
        e = rates(acc, m)
        assert abs(e['hidden'] / e['visible'] - 1) < 0.05, e                 # no dependence -> ratio 1
        e = states(acc, m)
        assert abs(e['hidden'] / e['train'] - 1) < 1e-6 and abs(e['shown'] / e['train'] - 1) < 1e-6, e
    row = CC.summarise(acc)
    args = ('yes', 'some', 'no')
    assert CC.verdict([row['h']['modality']['ratio']['all']] * 3, [1.0], *args).startswith('no')
    assert CC.verdict([1.6, 1.7], [1.2], *args).startswith('yes')
    assert CC.verdict([1.6, 1.7], [0.99], *args).startswith('some')          # a single model at or below 1
    assert CC.verdict([1.2, 1.7], [1.1], *args).startswith('some')
    assert CC.verdict([1.05, 1.7], [1.01], *args).startswith('no')


def test_same_masks_for_every_model():
    a = CC.measure(Copy(), sampler, torch.arange(128), 64)
    b = CC.measure(Copy(), sampler, torch.arange(128), 64)
    c = CC.measure(Normed(), sampler, torch.arange(128), 64, 'training')
    for m in 'hl':
        assert all((a[m]['position'][k] == b[m]['position'][k]).all() for k in CC.GROUPS)
        assert all(a[m]['position'][k][2] == c[m]['position'][k][2] for k in CC.GROUPS)      # the same positions
        assert all(a[m]['modality'][s][x][1] == c[m]['modality'][s][x][1] for s in CC.STATES for x in CC.BINS)


def test_batch_statistics():
    bn = torch.nn.BatchNorm2d(3)
    bn.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    bn.running_var.fill_(4.0)
    bn.eval()
    before = {k: v.clone() for k, v in bn.state_dict().items()}
    x = 7 + 3 * torch.randn(64, 3, 5, 5, generator=torch.Generator().manual_seed(1))
    y = 2 * x[:16] - 1                                                        # another batch, normalised like x
    with CC.batch_statistics(bn) as adopt:
        a = adopt(lambda: bn(x))                                             # with the statistics of x
        assert not bn.training
        assert a.mean((0, 2, 3)).abs().max() < 1e-5 and (a.var((0, 2, 3), unbiased=False) - 1).abs().max() < 1e-4
        assert (bn(x) - a).abs().max() < 1e-5                                # the passes that follow keep them exactly
        assert (bn(y) - standardise(y, x)).abs().max() < 1e-4
    after = bn.state_dict()
    assert all((before[k] == after[k]).all() for k in before) and bn.momentum == 0.1 and not bn.training
    with CC.batch_statistics(bn, False) as adopt:                             # not active: the running statistics
        assert (adopt(lambda: bn(x)) - (x - before['running_mean'][None, :, None, None]) / 2).abs().max() < 1e-3
    assert all((before[k] == bn.state_dict()[k]).all() for k in before)


def standardise(x, ref, eps=1e-5):
    """x normalised per channel with the statistics of the batch ref, as a BatchNorm layer in training mode does."""
    mean, var = ref.mean((0, 2, 3), keepdim=True), ref.var((0, 2, 3), unbiased=False, keepdim=True)
    return (x - mean) / (var + eps).sqrt()


def test_normalisation_of_training():
    model, idx = Normed(), torch.arange(64)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    test = CC.measure(model, sampler, idx, 64, 'test')
    train = CC.measure(model, sampler, idx, 64, 'training')
    assert all((before[k] == v).all() for k, v in model.state_dict().items()) and not model.bn['h'].training
    # by hand, for the one batch: the targets with the statistics of the clean batch, the masked batch with its own,
    # and the states of test 2 with the statistics of the batch masked as in training
    h, l = sampler(idx)
    mh, ml = model.draw_masks(64, torch.Generator().manual_seed(0))
    hm, lm = model.fill(h, l, mh, ml)
    tokens = lambda a, b: torch.stack([a.flatten(1), b.flatten(1)], 2)       # noqa: E731
    clean = tokens(standardise(h, h), standardise(l, l))
    error = lambda z: float((((z - clean) ** 2).mean(2) * mh).sum() / mh.sum())   # noqa: E731
    want = {'train': error(tokens(standardise(hm, hm), standardise(lm, lm))),
            'shown': error(tokens(standardise(hm, hm), standardise(l, lm))),
            'hidden': error(tokens(standardise(hm, hm), standardise(torch.zeros_like(l), lm)))}
    got = states(train, 'h')
    assert all(abs(got[s] - want[s]) < 1e-4 * want[s] for s in CC.STATES), (got, want)
    assert abs(got['shown'] - got['train']) < 1e-5 * got['train'], got       # the LiDAR is there in both states
    # as at test time the same model normalises with its running statistics (mean 5, variance 9)
    run = lambda x: (x - 5) / (9 + 1e-5) ** 0.5                              # noqa: E731
    clean = tokens(run(h), run(l))
    got = states(test, 'h')
    assert abs(got['train'] - error(tokens(run(hm), run(lm)))) < 1e-4 * got['train'], got
    assert abs(got['train'] / want['train'] - 1) > 0.5                       # the two normalisations differ here


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print('test_completion_check: %d tests passed' % len(tests))
