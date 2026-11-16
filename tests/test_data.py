"""Unit tests of symbionet/data.py. The checks on the real scenes run only when the dataset folder is found
(environment variable SYMBIONET_DATA, or ./dataset next to the repository); the others use synthetic images.

Run with `python tests/test_data.py` (prints a summary) or with pytest.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import symbionet.data as D  # noqa: E402

DATA = Path(os.environ.get('SYMBIONET_DATA', ROOT / 'dataset'))
# per-class training pixels of the 1 % split in Geo-DeltaSSM (Tables I-III): the counts our splits must have
PAPER_TRAIN = {'trento': [41, 30, 5, 92, 106, 32],
               'houston2013': [13, 13, 7, 13, 13, 4, 13, 13, 13, 13, 13, 13, 5, 5, 7],
               'muufl': [233, 43, 69, 19, 67, 5, 23, 63, 14, 2, 3]}


def synthetic_scene(H=13, W=17, C=5, K=3, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(-1, K, (H, W))
    labels[0, :K] = np.arange(K)                              # every class present
    return D.Scene('synthetic', rng.standard_normal((C, H, W)).astype(np.float32),
                   rng.standard_normal((2, H, W)).astype(np.float32), labels, np.ones((H, W), bool),
                   ['c%d' % k for k in range(K)])


def test_patches_equal_padded_slices():
    s = synthetic_scene()
    for p, padding in [(p, m) for p in (3, 5, 7, 9) for m in ('zero', 'reflect')]:
        sampler = D.PatchSampler(s, p, 'cpu', padding)
        r = p // 2
        np_mode = 'constant' if padding == 'zero' else 'reflect'
        hp = np.pad(s.hsi, ((0, 0), (r, r), (r, r)), mode=np_mode)
        lp = np.pad(s.lidar, ((0, 0), (r, r), (r, r)), mode=np_mode)
        H, W = s.shape
        idx = np.array([0, W - 1, (H - 1) * W, H * W - 1, 5 * W + 7, 3])           # corners and inside
        h, l = sampler(torch.from_numpy(idx))
        for b, i in enumerate(idx):
            y, x = divmod(int(i), W)
            assert np.array_equal(h[b].numpy(), hp[:, y:y + p, x:x + p]), (p, i)
            assert np.array_equal(l[b].numpy(), lp[:, y:y + p, x:x + p]), (p, i)
        assert torch.equal(sampler.targets(torch.from_numpy(idx)), torch.from_numpy(s.labels.ravel()[idx]))


def test_augmentation_is_shared_by_the_modalities():
    s = synthetic_scene()
    s.lidar = np.repeat(s.hsi[:1], 2, 0)                      # LiDAR = HSI band 0
    sampler = D.PatchSampler(s, 5, 'cpu')
    idx = torch.arange(40)
    for mode in ('flip', 'dihedral'):
        h, l = sampler(idx, mode, torch.Generator().manual_seed(0))
        assert torch.equal(h[:, :1], l[:, :1]) and torch.equal(l[:, 0], l[:, 1]), mode
    h0, _ = sampler(idx)
    inner = torch.full((200,), 6 * s.shape[1] + 8)           # an interior pixel (border patches are symmetric)
    hd, _ = sampler(inner, 'dihedral', torch.Generator().manual_seed(1))
    assert len({tuple(x.flatten().tolist()) for x in hd[:, 0]}) == 8        # all symmetries of the square
    hf, _ = sampler(inner, 'flip', torch.Generator().manual_seed(1))
    assert len({tuple(x.flatten().tolist()) for x in hf[:, 0]}) == 4        # flips only
    assert torch.equal(sampler(idx, 'none')[0], h0)


def test_exact_ceiling():
    assert D.n_train_per_class([1300, 1251, 183, 100, 99, 1], 0.01).tolist() == [13, 13, 2, 1, 1, 1]


def test_split_is_disjoint_and_complete():
    s = synthetic_scene(40, 50, K=4)
    sp = D.make_split(s.labels, 0.1, 3)
    D.check_split(s, sp)
    counts = np.bincount(s.labels[s.labels >= 0], minlength=4)
    assert D.class_counts(s, sp['train']).tolist() == D.n_train_per_class(counts, 0.1).tolist()
    assert np.array_equal(sp['train'], D.make_split(s.labels, 0.1, 3)['train'])       # same seed, same split
    assert not np.array_equal(sp['train'], D.make_split(s.labels, 0.1, 4)['train'])
    pool = D.pool_indices(s, sp)
    assert not np.intersect1d(pool, sp['test']).size and np.isin(sp['train'], pool).all()


def test_subset_train():
    s = synthetic_scene(40, 50, K=4)
    sp = D.make_split(s.labels, 0.3, 3)
    full = D.class_counts(s, sp['train'])
    for frac in (0.5, 0.25, 0.01):
        sub = D.subset_train(s, sp['train'], frac, seed=0)
        want = [max(1, int(np.ceil(n * frac - 1e-9))) for n in full]
        assert D.class_counts(s, sub).tolist() == want, frac                    # per class, at least one pixel
        assert np.isin(sub, sp['train']).all() and (np.diff(sub) > 0).all()
        assert np.array_equal(sub, D.subset_train(s, sp['train'], frac, seed=0))
    assert not np.array_equal(D.subset_train(s, sp['train'], 0.5, 0), D.subset_train(s, sp['train'], 0.5, 1))
    assert D.subset_train(s, sp['train'], 1.0) is sp['train']
    left = np.setdiff1d(sp['train'], D.subset_train(s, sp['train'], 0.5, 0))
    assert np.isin(left, D.pool_indices(s, sp)).all()                           # the pixels left out stay in the pool
    for bad in (0, -0.5, 1.5):
        try:
            D.subset_train(s, sp['train'], bad)
            raise AssertionError('train_fraction %s was accepted' % bad)
        except ValueError:
            pass


def test_pool_options():
    s = synthetic_scene(40, 50, K=4)
    sp = D.make_split(s.labels, 0.1, 3)
    pool = D.pool_indices(s, sp)
    valid = np.flatnonzero(s.valid.ravel())
    assert np.array_equal(pool, np.setdiff1d(valid, sp['test']))                # the default: no test pixel
    assert np.array_equal(D.pool_indices(s, sp, False, 0, 0), pool)
    assert np.array_equal(D.pool_indices(s, sp, with_test=True), valid)         # every valid pixel
    sub = D.pool_indices(s, sp, size=100, seed=0)
    assert len(sub) == 100 and len(np.unique(sub)) == 100 and np.isin(sub, pool).all() and (np.diff(sub) > 0).all()
    assert np.array_equal(sub, D.pool_indices(s, sp, size=100, seed=0))         # the same subset every time
    assert not np.array_equal(sub, D.pool_indices(s, sp, size=100, seed=1))
    assert np.array_equal(D.pool_indices(s, sp, size=len(pool) + 5), pool)      # not smaller than the pool: all of it
    both = D.pool_indices(s, sp, with_test=True, size=200, seed=0)
    assert len(both) == 200 and np.isin(both, valid).all() and np.intersect1d(both, sp['test']).size > 0
    try:
        D.pool_indices(s, sp, size=-1)
        raise AssertionError('a negative pool size was accepted')
    except ValueError:
        pass


def test_normalise_and_nodata():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((20, 30, 4)) * [1, 5, 10, 100] + [0, 3, -2, 50]).astype(np.float32)
    x[0, :5] = 0                                              # no-data: zero spectra
    x[1, :] = 7.0                                             # no-data: one fill spectrum on 60 pixels ...
    x[2, :] = 7.0
    nod = D.nodata_mask(x)
    assert nod[0, :5].all() and nod[1].all() and nod[2].all() and nod.sum() == 65
    z = D.normalise(x, ~nod, 'zscore')
    assert np.allclose(z[~nod].mean(0), 0, atol=1e-4) and np.allclose(z[~nod].std(0), 1, atol=1e-4)
    m = D.normalise(x, ~nod, 'minmax')
    assert np.allclose(m[~nod].min(0), 0, atol=1e-6) and np.allclose(m[~nod].max(0), 1, atol=1e-6)
    p = D.pca(z, ~nod, 3, 'first')
    assert p.shape == (20, 30, 3) and abs(float(p[~nod][:, 0].std()) - 1) < 1e-4
    assert np.all(p[~nod].std(0)[1:] <= p[~nod].std(0)[0] + 1e-4)
    p = D.pca(z, ~nod, 3, 'each')
    assert np.allclose(p[~nod].std(0), 1, atol=1e-4) and np.allclose(p[~nod].mean(0), 0, atol=1e-4)


def test_real_scenes_match_the_paper():
    if not DATA.exists():
        print('   (skipped: no dataset folder at %s)' % DATA)
        return
    for name, expect in PAPER_TRAIN.items():
        s = D.load_scene(name, DATA)
        for kind in ('geo', 'random'):                       # the shipped Geo-DeltaSSM split and a new random one
            sp = D.get_split(s, ROOT / 'splits', kind) if kind == 'geo' else D.make_split(s.labels, 0.01, 0)
            D.check_split(s, sp)
            assert D.class_counts(s, sp['train']).tolist() == expect, (name, kind)
        assert np.isfinite(s.hsi).all() and np.isfinite(s.lidar).all(), name
        assert abs(float(s.hsi.min())) < 1e-6 and abs(float(s.hsi.max()) - 1) < 1e-6, name     # min-max per band


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
    print('test_data: %d/%d passed' % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
