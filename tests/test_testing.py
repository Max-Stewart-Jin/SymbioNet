"""Unit tests of symbionet/testing.py and of the test stages of formal.py (synthetic data, no dataset needed).

Run with `python tests/test_testing.py` (prints a summary) or with pytest.
"""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal as F  # noqa: E402
import symbionet.conditions as K  # noqa: E402
import symbionet.data as D  # noqa: E402
import symbionet.testing as T  # noqa: E402
from symbionet.config import apply  # noqa: E402
from symbionet.corrupt import Condition  # noqa: E402


def trained(tmp: Path, seeds=(0, 1)):
    """Two short runs of the full method on a small synthetic scene: (scene, split, cfg, records)."""
    from symbionet.runs import run_one, write_config
    rng = np.random.default_rng(0)
    H, W, Kc = 20, 24, 3
    labels = rng.integers(-1, Kc, (H, W))
    labels[0, :Kc] = np.arange(Kc)
    scene = D.Scene('synthetic', rng.random((6, H, W)).astype(np.float32), rng.random((1, H, W)).astype(np.float32),
                    labels, np.ones((H, W), bool), ['c%d' % k for k in range(Kc)])
    cfg = apply(F.Plan().config('trento', 'full'), [('data.patch', 5), ('training.epochs', 2), ('eval.every', 1),
                                                     ('training.batch_size', 16), ('training.pool_batch', 16)])

    class St:                                            # what runs.Setting holds, for a scene that is not on disk
        pass
    st = St()
    st.dataset, st.scene = 'synthetic', scene
    st.split = dict(D.make_split(scene.labels, 0.3, 0), path='synthetic.npz')
    st.pool = D.pool_indices(scene, st.split)
    st.sampler = D.PatchSampler(scene, 5, torch.device('cpu'), 'zero')
    write_config(tmp, cfg)
    records = {s: run_one(cfg, st, s, s, torch.device('cpu'), tmp, 'full', log=lambda *_: None) for s in seeds}
    return scene, st.split, cfg, records


def test_bench_repeats_the_training_records_and_scores_conditions():
    tmp = Path(tempfile.mkdtemp(prefix='testing_'))
    try:
        scene, split, cfg, records = trained(tmp)
        bench = T.Bench(scene, split, cfg['data'], torch.device('cpu'), chunk=64, block=4)
        assert bench.header() == dict(dataset='synthetic', n_test=len(split['test']), split='synthetic.npz', seed=0,
                                      block=4)
        members = [T.Member('full/%s' % w, s, cfg, tmp / 'ckpt' / ('run%d_%s.pt' % (s, w)), records[s][w]['OA'])
                   for w in ('best', 'last') for s in records]
        assert all(bench.fits(m) == [] for m in members)
        for m in members:
            bench.load(m)
        preds = {}
        clean = bench.score(members, None, on_clean=lambda m, p: preds.__setitem__((m.key, m.seed), p))
        for m, s in zip(members, clean):                      # the clean scores are those of the training record
            assert abs(s['OA'] - m.recorded) < 1e-9 and set(s) == set(T.KEEP), (m.key, m.seed)
            assert preds[(m.key, m.seed)].shape == bench.y.shape
            assert abs(100 * (preds[(m.key, m.seed)] == bench.y).mean() - s['OA']) < 1e-9
        one = bench.score(members[:1], Condition('pixel', 'lidar', 0.5))[0]
        assert 0 <= one['OA'] <= 100 and 'OA_per_draw' not in one
        blk = bench.score(members[:2], Condition('block', 'hsi', 0.3))
        assert all(len(s['OA_per_draw']) == 3 and abs(np.mean(s['OA_per_draw']) - s['OA']) < 1e-9 for s in blk)
        sh = bench.score(members[:1], K.Shift(1.0))[0]
        assert len(sh['OA_per_draw']) == 4
        again = bench.score(members[:1], K.Shift(1.0))[0]
        assert again == sh                                    # the same test scenes every time
        near = bench.score(members[:1], K.Refill(Condition('block', 'lidar', 0.3)))[0]
        assert len(near['OA_per_draw']) == 3
        # the protocol of the 46 conditions: the same numbers as the loop of robustness.py
        from symbionet.corrupt import TESTS, band_stats, corrupt_scene
        from symbionet.engine import evaluate
        stats = {m: band_stats(getattr(scene, m), scene.valid) for m in ('hsi', 'lidar')}
        idx = torch.from_numpy(split['test'])
        for cond in (Condition('noise', 'both', 10.0), Condition('block', 'lidar', 0.3), Condition('bands', 'hsi', 0.5),
                     Condition('missing', 'hsi', 1.0)):
            new = bench.score(members[:2], cond)
            for m, n in zip(members[:2], new):
                old = [evaluate(m.model, D.PatchSampler(corrupt_scene(scene, cond, stats, 0, 4, d), 5,
                                                        torch.device('cpu'), 'zero'), idx, scene.n_classes, 64)
                       for d in range(TESTS[cond.test]['draws'])]
                assert all(abs(np.mean([o[k] for o in old]) - n[k]) < 1e-9 for k in T.KEEP), cond.label
        # a model that was trained with another split or normalisation does not fit the bench
        other = T.Member('x/best', 0, dict(cfg, data=dict(cfg['data'], lidar_norm='zscore')), members[0].ckpt)
        assert bench.fits(other) == ['lidar_norm']
        # a part of the test pixels, as training uses with eval.subset
        part = T.Bench(scene, split, cfg['data'], torch.device('cpu'), subset=50)
        assert len(part.test) == 50 and set(part.test) <= set(split['test']) and part.header()['n_test'] == 50
    finally:
        shutil.rmtree(tmp)


def test_store_keeps_what_was_scored():
    tmp = Path(tempfile.mkdtemp(prefix='store_'))
    try:
        head = dict(dataset='d', n_test=10, split='s.npz', seed=0, block=16)
        path = tmp / 'tests' / 'd.json'
        assert T.Store.read(path) is None
        st = T.Store(path, head)
        assert not st.has('full/best', 'clean', 0)
        st.put('full/best', 'clean', 0, dict(OA=90.0, AA=80.0, Kappa=70.0, F1=60.0))
        st.put('full/best', 'clean', 1, dict(OA=92.0, AA=80.0, Kappa=70.0, F1=60.0))
        st.put('full/best', 'noise hsi 10', 0, dict(OA=50.0, AA=40.0, Kappa=30.0, F1=20.0))
        st.put_zones('full/best', 0, dict(shadow=80.0))
        st.describe('full/best', version='full', weights='best')
        st.save()
        assert not list(path.parent.glob('*.tmp'))
        back = T.Store(path, head)                             # the same setting: goes on where it stopped
        assert back.has('full/best', 'clean', 1) and not back.has('full/best', 'noise hsi 10', 1)
        assert back.get('full/best', 'clean') == {0: dict(OA=90.0, AA=80.0, Kappa=70.0, F1=60.0),
                                                  1: dict(OA=92.0, AA=80.0, Kappa=70.0, F1=60.0)}
        assert back.zones('full/best') == {0: dict(shadow=80.0)} and back.get('base/best', 'clean') == {}
        assert back.data['models']['full/best'] == dict(version='full', weights='best')
        assert T.mean_over_seeds(back, 'full/best', ['clean'])[0] == 91.0
        # only the seeds that have every condition count
        m, values, seeds = T.mean_over_seeds(back, 'full/best', ['clean', 'noise hsi 10'])
        assert (m, values, seeds) == (70.0, [70.0], [0])
        assert T.mean_over_seeds(back, 'full/best', ['clean'], seeds=[1])[0] == 92.0
        assert T.mean_over_seeds(back, 'full/best', ['missing lidar 1']) == (None, [], [])
        try:
            T.Store(path, dict(head, n_test=11))
            raise AssertionError('scores of another test setting were accepted')
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_zone_scores():
    #            shadow (zone 2)          outside (zone 0)             uncertain
    y = np.array([0, 0, 0, 0, 1, 1,       0, 0, 1, 1, 1, 1, 2, 2,      0])
    pred = np.array([0, 0, 0, 1, 1, 0,    0, 0, 1, 1, 1, 0, 2, 0,      1])
    zone = np.array([2] * 6 + [0] * 8 + [1])
    s = T.zone_scores(pred, y, zone, 3)
    assert abs(s['shadow'] - 100 * 4 / 6) < 1e-9 and abs(s['outside'] - 100 * 6 / 8) < 1e-9 and s['uncertain'] == 0.0
    assert abs(s['all'] - 100 * 10 / 15) < 1e-9
    # outside*: classes 0 and 1 of the shadow (4 and 2 pixels), with their accuracy outside (100 % and 75 %)
    assert abs(s['outside*'] - (4 * 100 + 2 * 75) / 6) < 1e-9
    assert abs(s['cost'] - (s['outside*'] - s['shadow'])) < 1e-9
    assert s['class_in'] == [75.0, 50.0, None] and s['class_out'] == [100.0, 75.0, 50.0]


def fake_root():
    tmp = Path(tempfile.mkdtemp(prefix='formal_tests_'))
    shutil.copytree(ROOT / 'configs', tmp / 'configs')
    return tmp, F.Plan(tmp / 'configs' / 'formal.yaml', tmp)


def fill(plan, dataset, values, seeds=(0, 1, 2, 3, 4), zones=None):
    """A store with the same score for every seed: values = {key: {label: OA}}; zones = {key: [shadow OA per seed]}."""
    st = T.Store(plan.store_path(dataset), dict(dataset=dataset, n_test=100, split='s.npz', seed=0, block=16))
    for key, per in values.items():
        for label, oa in per.items():
            for s in seeds:
                st.put(key, label, s, dict(OA=float(oa), AA=0.0, Kappa=0.0, F1=0.0))
    for key, shadow in (zones or {}).items():
        for s, v in enumerate(shadow):
            st.put_zones(key, s, {'all': 90.0, 'shadow': float(v), 'outside': 91.0, 'outside*': 92.0, 'cost': 1.0})
    st.save()


def test_the_test_stage_of_the_plan():
    plan = F.Plan()
    parts = {p['name']: p for p in plan.parts('tests1')}
    assert list(parts) == ['corruptions', 'extra', 'last']
    assert len(parts['corruptions']['conds']) == 46 and len(parts['corruptions']['versions']) == 9
    assert parts['corruptions']['weights'] == 'best' and parts['last']['weights'] == 'last'
    assert len(parts['extra']['conds']) == 13 and parts['extra']['versions'] == [
        'base', 'copies', 'copies_whole', 'copies_m2c', 'full']
    assert parts['last']['versions'] == ['base', 'full']
    assert plan.is_test('tests1') and not plan.is_test('stage1')
    assert plan.store_path('muufl') == ROOT / 'results' / 'formal' / 'tests' / 'muufl.json'
    assert (ROOT / plan.tests('tests1')['zones']['houston2013']).exists()
    # --smoke: the first and the last level of every test and target
    smoke = {p['name']: [c.label for c in p['conds']] for p in F.Plan(smoke=True).parts('tests1')}
    assert smoke['extra'] == ['shift lidar 1', 'shift lidar 3', 'pixel lidar 0.1 nearest', 'pixel lidar 0.9 nearest',
                              'block lidar 0.1 nearest', 'block lidar 0.5 nearest']
    assert len(smoke['corruptions']) == 20 and 'missing hsi 1' in smoke['corruptions']
    # what a test stage counts as work
    work = [(None, [1, 2]), (Condition('block', 'hsi', 0.1), [1, 2, 3]), (K.Shift(1.0), [1])]
    assert F._evals(work) == 2 + 3 * 3 + 4


def test_summary_and_checks_of_a_test_stage():
    tmp, plan = fake_root()
    try:
        text = F.tests_summary(plan, 'tests1')
        assert 'houston2013: nothing yet' in text and '0 of 8805 scores on disk.' in text
        assert 'NOT EVALUATED' in text
        corr = [c.label for c in K.suite('corruptions')]
        near = [c.label for c in K.suite('interp')]
        base = dict({lb: 70.0 for lb in corr}, clean=94.0)
        full = dict({lb: 88.0 for lb in corr}, clean=94.5)
        # B = full, A = base. Shift of 1 px: full drops 1.0, base 0.8 -> +0.2. Imputation: base with interpolation
        # reaches 80, full with the band means has 88 -> +8, and full with interpolation 85 -> +5.
        extra_b = dict({'shift lidar 1': 93.5, 'shift lidar 2': 92.0, 'shift lidar 3': 90.0}, **{lb: 85.0 for lb in near})
        extra_a = dict({'shift lidar 1': 93.2, 'shift lidar 2': 92.0, 'shift lidar 3': 90.0}, **{lb: 80.0 for lb in near})
        for d in ('houston2013', 'muufl', 'trento'):
            fill(plan, d, {'full/best': dict(full, **extra_b), 'base/best': dict(base, **extra_a),
                           'full/last': dict(full, clean=94.0), 'base/last': base},
                 zones={'full/best': [77, 83, 80, 76, 84], 'base/best': [83.2, 83.8, 83.5, 83.3, 83.7]}
                 if d == 'houston2013' else None)
        text = F.tests_summary(plan, 'tests1')
        row = [ln for ln in text.split('\n') if ln.startswith('full ') and '5/5' in ln][0]
        assert ' 94.50' in row and '88.00 ± 0.00' in row and '-6.50' in row, row
        assert 'houston2013 +0.20, muufl +0.20, trento +0.20' in text
        assert 'HOLDS: B is not more sensitive to the shift than A' in text
        assert text.count('HOLDS: B with the band means is better than A with interpolation') == 2   # pixel, block
        assert 'both with interpolation (the gate does not respond), B minus A: houston2013 +5.00' in text
        # shadow: -3.50 with a standard error of 1.59 (the two variances over five runs each): beyond two of them
        assert 'B minus A: -3.50 (standard error 1.59) -> B IS CLEARLY WORSE IN THE SHADOW' in text
        done, total = F._planned(plan, 'tests1')
        assert total == 8805 and 0 < done < total
    finally:
        shutil.rmtree(tmp)


def test_the_rules_decide_as_written():
    tmp, plan = fake_root()
    try:
        corr = [c.label for c in K.suite('corruptions')]
        near = [c.label for c in K.suite('interp')]

        def write(shift_b, near_a, shadow_b):
            for d in ('houston2013', 'muufl', 'trento'):
                p = plan.store_path(d)
                if p.exists():
                    p.unlink()
                fill(plan, d, {'full/best': dict({lb: 88.0 for lb in corr}, clean=94.0,
                                                 **{'shift lidar 1': shift_b[d]}, **{lb: 85.0 for lb in near}),
                               'base/best': dict({lb: 70.0 for lb in corr}, clean=94.0,
                                                 **{'shift lidar 1': 93.0}, **{lb: near_a[d] for lb in near})},
                     zones={'full/best': shadow_b, 'base/best': [83.0, 84.0, 83.5, 83.2, 83.8]}
                     if d == 'houston2013' else None)
            return F.tests_summary(plan, 'tests1')

        same = dict(houston2013=93.0, muufl=93.0, trento=93.0)
        # shift: +0.6 on one dataset only -> no claim; above +1 on two -> fails
        t = write(dict(same, muufl=92.4), dict(houston2013=80, muufl=80, trento=80), [83.0, 84.0, 83.5, 83.2, 83.8])
        assert 'houston2013 +0.00, muufl +0.60, trento +0.00' in t and '-> NO CLAIM' in t
        assert 'NO CLEAR DIFFERENCE' in t
        t = write(dict(houston2013=91.5, muufl=91.0, trento=93.0), dict(houston2013=80, muufl=80, trento=80),
                  [79.0, 80.0, 79.5, 79.2, 79.8])
        assert 'FAILS: B is more sensitive to the shift than A' in t
        assert 'B minus A: -4.00' in t and 'B IS CLEARLY WORSE IN THE SHADOW' in t
        # imputation: full with the band means has 88; base with interpolation 87.5 / 86 / 86 -> holds (two at +2,
        # none below -0.5); 89 on one dataset -> no claim; 89.5 on two -> fails
        t = write(same, dict(houston2013=87.5, muufl=86, trento=86), [89.0, 90.0, 89.5, 89.2, 89.8])
        assert t.count('HOLDS: B with the band means is better') == 2 and 'B IS CLEARLY BETTER IN THE SHADOW' in t
        t = write(same, dict(houston2013=89, muufl=86, trento=86), [83.0] * 5)
        assert t.count('-> NO CLAIM') == 2
        t = write(same, dict(houston2013=89.5, muufl=89.5, trento=80), [83.0] * 5)
        assert t.count('FAILS: A with interpolation is better than B with the band means') == 2
    finally:
        shutil.rmtree(tmp)


def test_pack_leaves_the_weights_out():
    tmp, plan = fake_root()
    try:
        out = plan.folder('trento', 'full')
        (out / 'ckpt').mkdir(parents=True)
        (out / 'ckpt' / 'run0_best.pt').write_bytes(b'weights')
        (out / 'run0.json').write_text('{}')
        (plan.out_root / 'running.lock').write_text('')
        fill(plan, 'trento', {'full/best': {'clean': 99.0}})
        z = F.pack(plan)
        assert z == tmp / 'results' / 'SymbioNet_formal_results.zip'
        names = sorted(zipfile.ZipFile(z).namelist())
        assert names == ['formal/tests/trento.json', 'formal/trento/full/run0.json'], names
    finally:
        shutil.rmtree(tmp)


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print('PASS', name)
        except Exception as e:                  # noqa: BLE001 - report every failure, keep going
            failed.append(name)
            print('FAIL', name, '-', type(e).__name__, e)
    print('=' * 60)
    print('test_testing: %d/%d passed' % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
