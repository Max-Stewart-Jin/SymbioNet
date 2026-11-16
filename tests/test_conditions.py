"""Unit tests of symbionet/conditions.py (synthetic images). Run with `python tests/test_conditions.py` or with
pytest."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import symbionet.conditions as K  # noqa: E402
import symbionet.corrupt as C  # noqa: E402
import symbionet.data as D  # noqa: E402


def scene(H=60, W=80, Ch=12, Cl=1, seed=0, holes=False):
    rng = np.random.default_rng(seed)
    hsi = (rng.random((Ch, H, W)) * np.linspace(0.5, 2, Ch)[:, None, None]).astype(np.float32)
    lidar = rng.random((Cl, H, W)).astype(np.float32)
    labels = rng.integers(-1, 3, (H, W))
    valid = np.ones((H, W), bool)
    if holes:
        valid[10:20, 30:40] = False                               # a no-data area
    return D.Scene('synthetic', hsi, lidar, labels, valid, ['a', 'b', 'c'])


def stats_of(s):
    return {m: C.band_stats(getattr(s, m), s.valid) for m in ('hsi', 'lidar')}


def test_the_suites():
    corr = K.suite('corruptions')
    assert len(corr) == 46 and corr == C.conditions(K.PAPER_TESTS)
    assert sum(K.n_draws(c) for c in corr) == 82                  # the corrupted images of the paper's protocol
    assert [c.label for c in K.suite('shift')] == ['shift lidar 1', 'shift lidar 2', 'shift lidar 3']
    interp = K.suite('interp')
    assert len(interp) == 10 and all(c.target == 'lidar' for c in interp)
    assert interp[0].label == 'pixel lidar 0.1 nearest' and interp[-1].label == 'block lidar 0.5 nearest'
    assert K.n_draws(K.suite('shift')[0]) == 4 and K.n_draws(interp[0]) == 1 and K.n_draws(interp[-1]) == 3
    labels = [c.label for s in K.SUITES for c in K.suite(s)]
    assert len(set(labels)) == len(labels) == 59                  # a label names one condition
    for bad in (lambda: K.suite('nothing'), lambda: K.Refill(C.Condition('noise', 'lidar', 10.0)),
                lambda: K.Refill(C.Condition('pixel', 'lidar', 0.1), 'linear')):
        try:
            bad()
            raise AssertionError('accepted')
        except ValueError:
            pass


def test_corrupt_py_is_untouched_by_the_new_conditions():
    """The conditions of corrupt.py give the same image through apply() as through corrupt_scene()."""
    s = scene()
    st = stats_of(s)
    for cond in (C.Condition('pixel', 'both', 0.5), C.Condition('block', 'hsi', 0.3), C.Condition('noise', 'lidar', 10.0),
                 C.Condition('bands', 'hsi', 0.5), C.Condition('missing', 'lidar', 1.0)):
        a, b = K.apply(s, cond, st, 3, 8, 1), C.corrupt_scene(s, cond, st, 3, 8, 1)
        assert np.array_equal(a.hsi, b.hsi) and np.array_equal(a.lidar, b.lidar), cond.label


def test_shift_moves_the_lidar_only():
    s = scene()
    st = stats_of(s)
    x = s.lidar
    for k in (1, 3):
        up, down, left, right = (K.apply(s, K.Shift(float(k)), st, draw=d) for d in range(4))
        for out in (up, down, left, right):
            assert out.hsi is s.hsi and out.labels is s.labels and out.lidar.shape == x.shape
        assert np.array_equal(down.lidar[:, k:], x[:, :-k])           # the content moved down by k rows
        assert all(np.array_equal(down.lidar[:, r], x[:, 0]) for r in range(k))      # the border repeats row 0
        assert np.array_equal(up.lidar[:, :-k], x[:, k:])
        assert all(np.array_equal(up.lidar[:, -1 - r], x[:, -1]) for r in range(k))
        assert np.array_equal(right.lidar[:, :, k:], x[:, :, :-k])
        assert all(np.array_equal(right.lidar[:, :, c], x[:, :, 0]) for c in range(k))
        assert np.array_equal(left.lidar[:, :, :-k], x[:, :, k:])
    assert np.array_equal(K.shift_image(x, 0, 0), x)
    assert np.array_equal(x, s.lidar)                                 # the input is not modified
    try:
        K.apply(s, K.Shift(1.5), st)
        raise AssertionError('a shift by half a pixel was accepted')
    except ValueError:
        pass


def test_lost_pixels_are_those_of_corrupt_py():
    s = scene()
    st = stats_of(s)
    for cond, draw in ((C.Condition('pixel', 'lidar', 0.3), 0), (C.Condition('pixel', 'both', 0.7), 0),
                       (C.Condition('block', 'lidar', 0.4), 2), (C.Condition('block', 'hsi', 0.2), 1)):
        lost = K.lost_pixels(s, cond, seed=5, block=8, draw=draw)
        out = C.corrupt_scene(s, cond, st, 5, 8, draw)
        for m in ('hsi', 'lidar'):
            changed = (getattr(out, m) != getattr(s, m)).any(0)
            assert np.array_equal(lost[m], changed), (cond.label, m)   # exactly the pixels that corrupt.py fills
        assert lost['lidar'].any() == (cond.target in ('lidar', 'both'))
    try:
        K.lost_pixels(s, C.Condition('noise', 'lidar', 10.0))
        raise AssertionError('accepted')
    except ValueError:
        pass


def test_nearest_fill():
    x = np.arange(2 * 5 * 6, dtype=np.float32).reshape(2, 5, 6)
    lost = np.zeros((5, 6), bool)
    lost[2, 2] = lost[2, 3] = lost[0, 0] = True
    valid = np.ones((5, 6), bool)
    y = K.nearest_fill(x, lost, valid)
    assert np.array_equal(y[:, ~lost], x[:, ~lost]) and np.array_equal(x[:, 2, 2], [14, 44])   # input untouched
    for r, c in ((2, 2), (2, 3), (0, 0)):
        near = [(rr, cc) for rr in range(5) for cc in range(6)
                if not lost[rr, cc] and (rr - r) ** 2 + (cc - c) ** 2 == 1]
        assert any(np.array_equal(y[:, r, c], x[:, rr, cc]) for rr, cc in near), (r, c)       # a direct neighbour
    valid[1, 2] = valid[3, 2] = valid[2, 1] = False                # no-data pixels are never a source
    y = K.nearest_fill(x, lost, valid)
    assert not any(np.array_equal(y[:, 2, 2], x[:, rr, cc]) for rr, cc in ((1, 2), (3, 2), (2, 1)))
    try:
        K.nearest_fill(x, np.ones((5, 6), bool), valid)
        raise AssertionError('accepted')
    except ValueError:
        pass


def test_refill_changes_only_the_lost_pixels():
    s = scene(holes=True)
    st = stats_of(s)
    for base, draw in ((C.Condition('pixel', 'lidar', 0.5), 0), (C.Condition('block', 'lidar', 0.3), 1)):
        mean_filled = C.corrupt_scene(s, base, st, 0, 8, draw)
        near = K.apply(s, K.Refill(base), st, 0, 8, draw)
        lost = (mean_filled.lidar != s.lidar).any(0)
        assert near.hsi is s.hsi
        assert np.array_equal(near.lidar[:, ~lost], s.lidar[:, ~lost])
        assert not np.allclose(near.lidar[:, lost], st['lidar'][0][:, None])          # not the band means
        values = set(s.lidar[0][~lost & s.valid].tolist())
        assert all(v in values for v in near.lidar[0][lost].tolist())                 # copies of pixels that are left
        again = K.apply(s, K.Refill(base), st, 0, 8, draw)
        assert np.array_equal(again.lidar, near.lidar)                                # repeatable


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
    print('test_conditions: %d/%d passed' % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
