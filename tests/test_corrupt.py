"""Unit tests of symbionet/corrupt.py (synthetic images). Run with `python tests/test_corrupt.py` or with pytest."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import symbionet.corrupt as C  # noqa: E402
import symbionet.data as D  # noqa: E402


def scene(H=60, W=80, Ch=12, Cl=2, seed=0):
    rng = np.random.default_rng(seed)
    hsi = (rng.random((Ch, H, W)) * np.linspace(0.5, 2, Ch)[:, None, None]).astype(np.float32)
    lidar = rng.random((Cl, H, W)).astype(np.float32)
    labels = rng.integers(-1, 3, (H, W))
    return D.Scene('synthetic', hsi, lidar, labels, np.ones((H, W), bool), ['a', 'b', 'c'])


def stats_of(s):
    return {m: C.band_stats(getattr(s, m), s.valid) for m in ('hsi', 'lidar')}


def test_noise_has_the_requested_snr():
    s = scene(200, 200)
    st = stats_of(s)
    for snr in (20, 10, 0):
        out = C.corrupt_scene(s, C.Condition('noise', 'hsi', snr), st)
        noise_var = (out.hsi - s.hsi).reshape(s.hsi.shape[0], -1).var(1)
        assert np.allclose(noise_var, st['hsi'][1] / 10 ** (snr / 10), rtol=0.03), snr
        assert out.lidar is s.lidar                              # the other modality is untouched


def test_pixel_loss_fraction_and_fill():
    s = scene()
    st = stats_of(s)
    out = C.corrupt_scene(s, C.Condition('pixel', 'both', 0.3), st)
    for m in ('hsi', 'lidar'):
        x, y = getattr(s, m), getattr(out, m)
        lost = (y != x).any(0)
        assert abs(lost.mean() - 0.3) < 0.03, m
        assert np.allclose(y[:, lost], st[m][0][:, None].astype(np.float32))
        assert np.array_equal(y[:, ~lost], x[:, ~lost])
    lost_h, lost_l = (out.hsi != s.hsi).any(0), (out.lidar != s.lidar).any(0)
    assert not np.array_equal(lost_h, lost_l)                     # 'both': independent masks


def test_block_coverage():
    s = scene(100, 120)
    st = stats_of(s)
    out = C.corrupt_scene(s, C.Condition('block', 'hsi', 0.25), st, block=16)
    lost = (out.hsi != s.hsi).any(0)
    assert 0.25 <= lost.mean() < 0.25 + 16 * 16 / lost.size
    assert np.allclose(out.hsi[:, lost], st['hsi'][0][:, None].astype(np.float32))


def test_bands_and_missing():
    s = scene()
    st = stats_of(s)
    out = C.corrupt_scene(s, C.Condition('bands', 'hsi', 0.5), st)
    changed = [(out.hsi[b] != s.hsi[b]).any() for b in range(s.hsi.shape[0])]
    assert sum(changed) == 6
    for b in np.flatnonzero(changed):
        assert np.allclose(out.hsi[b], np.float32(st['hsi'][0][b]))
    out = C.corrupt_scene(s, C.Condition('missing', 'lidar', 1), st)
    assert np.allclose(out.lidar, st['lidar'][0][:, None, None].astype(np.float32)) and out.hsi is s.hsi


def lost_pixels(s, out, m):
    return (getattr(out, m) != getattr(s, m)).any(0)


def test_levels_are_nested():
    """What is lost at a level is also lost at every higher level; the noise only changes its scale."""
    s = scene(100, 120)
    st = stats_of(s)
    for test, levels in [('pixel', (0.1, 0.3, 0.5, 0.9)), ('block', (0.1, 0.2, 0.4))]:
        masks = [lost_pixels(s, C.corrupt_scene(s, C.Condition(test, 'hsi', v), st), 'hsi') for v in levels]
        assert all((a <= b).all() and a.sum() < b.sum() for a, b in zip(masks, masks[1:])), test
    bands = [{b for b in range(s.hsi.shape[0]) if (C.corrupt_scene(s, C.Condition('bands', 'hsi', v), st).hsi[b]
                                                   != s.hsi[b]).any()} for v in (0.1, 0.3, 0.5, 0.7)]
    assert all(a < b for a, b in zip(bands, bands[1:]))

    def unit_noise(snr):                            # the noise divided by its standard deviation
        sigma = np.sqrt(st['hsi'][1] / 10 ** (snr / 10)).astype(np.float32)[:, None, None]
        return (C.corrupt_scene(s, C.Condition('noise', 'hsi', snr), st).hsi - s.hsi) / sigma

    z = [unit_noise(v) for v in (20, 0)]
    assert np.allclose(z[0], z[1], atol=1e-4)


def test_both_is_the_two_single_targets():
    """'both' corrupts each modality exactly as 'hsi' and 'lidar' do."""
    s = scene()
    st = stats_of(s)
    for test in ('noise', 'pixel'):
        both = C.corrupt_scene(s, C.Condition(test, 'both', 0.5 if test == 'pixel' else 10), st)
        hsi = C.corrupt_scene(s, C.Condition(test, 'hsi', 0.5 if test == 'pixel' else 10), st)
        lidar = C.corrupt_scene(s, C.Condition(test, 'lidar', 0.5 if test == 'pixel' else 10), st)
        assert np.array_equal(both.hsi, hsi.hsi) and np.array_equal(both.lidar, lidar.lidar), test


def test_draws_are_independent():
    s = scene()
    st = stats_of(s)
    cond = C.Condition('block', 'hsi', 0.3)
    d0, d1 = (lost_pixels(s, C.corrupt_scene(s, cond, st, draw=d), 'hsi') for d in (0, 1))
    assert not np.array_equal(d0, d1)
    assert np.array_equal(d0, lost_pixels(s, C.corrupt_scene(s, cond, st, draw=0), 'hsi'))
    assert {t: C.TESTS[t]['draws'] for t in C.TESTS} == {'noise': 1, 'pixel': 1, 'block': 3, 'bands': 5, 'missing': 1}


def test_reproducible_and_input_unchanged():
    s = scene()
    st = stats_of(s)
    before = s.hsi.copy(), s.lidar.copy()
    for cond in C.conditions():
        a = C.corrupt_scene(s, cond, st, seed=3, draw=2)
        b = C.corrupt_scene(s, cond, st, seed=3, draw=2)
        assert np.array_equal(a.hsi, b.hsi) and np.array_equal(a.lidar, b.lidar), cond.label
    assert np.array_equal(s.hsi, before[0]) and np.array_equal(s.lidar, before[1])
    a = C.corrupt_scene(s, C.Condition('noise', 'hsi', 10), st, seed=0)
    b = C.corrupt_scene(s, C.Condition('noise', 'hsi', 10), st, seed=1)
    assert not np.array_equal(a.hsi, b.hsi)


def test_condition_list():
    conds = C.conditions()
    assert len(conds) == 46 and len({c.label for c in conds}) == 46
    assert [c.test for c in C.conditions(['missing'])] == ['missing', 'missing']
    try:
        C.conditions(['blur'])
        raise AssertionError('an unknown test must be rejected')
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
    print('test_corrupt: %d/%d passed' % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
