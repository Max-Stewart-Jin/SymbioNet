"""Unit tests of formal.py and configs/formal.yaml (no training, no dataset needed).

Run with `python tests/test_formal.py` (prints a summary) or with pytest.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import formal as F  # noqa: E402
from symbionet.config import apply, dataset_config, parse_assignments  # noqa: E402

PATCH = {'houston2013': 11, 'muufl': 9, 'trento': 9}
# The command lines of the screening runs (11-07 to 11-12) that every version of the plan must repeat exactly.
MASKED = ('eval.every=5 model.decoder=identity model.rec_weighting=none model.lam=0 model.mask_mode=complementary '
          'model.fill=mean')
FULL = MASKED + ' model.mask_labeled=true model.pre_gate=known model.pre_gate_p=0.5'
SCREENING = {
    'base': 'eval.every=5',                                                           # tune_base
    'm2c': MASKED,                                                                    # lost/fillmean
    'copies': MASKED + ' model.decoder=null model.mask_labeled=true',                 # ctrl_pix
    'copies_whole': MASKED + ' model.decoder=null model.mask_block={patch} model.mask_labeled=true',   # ctrl_whole
    'copies_m2c': MASKED + ' model.mask_labeled=true',                                # comp_lab
    'copies_gate': MASKED + ' model.decoder=null model.mask_labeled=true model.pre_gate=known model.pre_gate_p=0.5',
    'full': FULL,                                                                     # pg_half
    'full_indep': FULL + ' model.mask_mode=independent',
    'full_same': FULL + ' model.mask_mode=same',
}


def fake_root(done=()):
    """A copy of the configs in a temporary project, with finished runs written for (dataset, version, seed, OA)."""
    tmp = Path(tempfile.mkdtemp(prefix='formal_test_'))
    shutil.copytree(ROOT / 'configs', tmp / 'configs')
    plan = F.Plan(tmp / 'configs' / 'formal.yaml', tmp)
    for d, v, s, oa in done:
        out = plan.folder(d, v)
        (out / 'ckpt').mkdir(parents=True, exist_ok=True)
        for w in ('best', 'last'):
            (out / 'ckpt' / ('run%d_%s.pt' % (s, w))).write_bytes(b'x')
        score = dict(OA=oa, AA=oa - 1, Kappa=oa - 2, F1=oa - 3, Precision=oa, per_class=[oa, oa])
        json.dump(dict(seed=s, best=score, last=dict(score, OA=oa - 0.5), best_epoch=100 + s, seconds=540.0,
                       device='cpu', code={'symbionet/model.py': 'abc'}, parameters=dict(test_time=1)),
                  open(out / ('run%d.json' % s), 'w'))
    return tmp, plan


def test_versions_repeat_the_screening_settings():
    plan = F.Plan()
    assert set(plan.stage('stage1')['versions']) == set(SCREENING), 'a version without a screening counterpart'
    for d, patch in PATCH.items():
        for v, line in SCREENING.items():
            pairs = parse_assignments(('data.patch=%d ' % patch + line.format(patch=patch)).split())
            want = apply(dataset_config(ROOT, d), pairs)
            assert plan.config(d, v) == want, (d, v, F.same_settings(plan.config(d, v), want))


def test_the_method_of_the_paper():
    m = F.Plan().config('houston2013', 'full')
    assert m['model']['decoder'] == 'identity' and m['model']['mask_labeled'] is True
    assert m['model']['pre_gate'] == 'known' and m['model']['pre_gate_p'] == 0.5
    assert m['model']['mask_mode'] == 'complementary' and m['model']['fill'] == 'mean' and m['model']['lam'] == 0
    assert m['model']['depth'] == 2 and m['model']['d'] == 64 and m['model']['band_mask'] == 0.0
    assert m['training']['epochs'] == 500 and m['eval']['every'] == 5 and m['eval']['subset'] == 0
    assert F.Plan().config('muufl', 'copies_whole')['model']['mask_block'] == 9
    assert F.Plan().config('houston2013', 'copies_whole')['model']['mask_block'] == 11


def test_stage1():
    plan = F.Plan()
    runs = plan.runs('stage1')
    assert len(runs) == 9 * 3 * 5 and len(set(runs)) == len(runs)
    assert runs[0] == ('houston2013', 'full', 0) and runs[5] == ('muufl', 'full', 0)
    assert [plan.kind('trento', v) for v in ('base', 'copies', 'copies_whole', 'copies_gate')] == [
        'base', 'copies', 'copies', 'copies']
    assert all(plan.kind('trento', v) == 'm2c' for v in ('full', 'm2c', 'copies_m2c', 'full_indep', 'full_same'))


def test_the_later_stages_change_only_what_they_name():
    """Fewer labels, parameter analysis and pool size: every version is full, base or copies with the named
    settings changed and nothing else."""
    plan = F.Plan()
    assert len(plan.runs('stage3')) == 6 * 3 * 5 and len(plan.runs('stage4')) == 15 * 3 * 3
    assert len(plan.runs('stage5')) == 2 * 1 * 5 and {s for _, _, s in plan.runs('stage4')} == {0, 1, 2}
    few = ['data.train_fraction', 'eval.every', 'training.epochs']
    changed = dict(full_half=few, base_half=few, copies_half=few, full_quarter=few, base_quarter=few,
                   copies_quarter=few, full_gate1=['model.pre_gate_p'], full_depth1=['model.depth'],
                   full_depth3=['model.depth'], full_wide96=['model.d', 'model.hidden'],
                   full_rho25=['model.mask_ratio'], full_rho375=['model.mask_ratio'],
                   full_alpha05=['training.masked_weight'], full_alpha2=['training.masked_weight'],
                   full_beta05=['model.beta'], full_beta2=['model.beta'], full_pool128=['training.pool_batch'],
                   full_pool512=['training.pool_batch'], full_patch_a=['data.patch'], full_patch_b=['data.patch'],
                   full_patch_c=['data.patch'], full_pool18k=['data.pool_size'], full_pool70k=['data.pool_size'])
    later = [v for st in ('stage3', 'stage4', 'stage5') for v in plan.stage(st)['versions']]
    assert sorted(later) == sorted(changed) and set(plan.raw['versions']) == set(SCREENING) | set(changed)
    for d in PATCH:
        for v, keys in changed.items():
            parent = v.split('_')[0]
            assert F.same_settings(plan.config(d, parent), plan.config(d, v)) == keys, (d, v)
    # the same number of training steps as with all labels, and about 100 evaluations per run
    steps = dict(houston2013=(5, 3, 2), muufl=(5, 3, 2), trento=(10, 5, 3))      # steps per epoch: all, half, quarter
    for d, (full, half, quarter) in steps.items():
        for v, per_epoch in (('full_half', half), ('full_quarter', quarter)):
            c = plan.config(d, v)
            assert 0 <= c['training']['epochs'] * per_epoch - 500 * full <= 10, (d, v)
            assert 95 <= c['training']['epochs'] / c['eval']['every'] <= 105, (d, v)
    assert plan.config('houston2013', 'full_patch_c')['data']['patch'] == 13
    assert plan.config('muufl', 'full_patch_a')['data']['patch'] == 5
    assert [p['versions'] for p in plan.parts('tests2')] == [plan.stage('stage3')['versions']]


def test_every_version_trains():
    """One epoch of every version of the plan on a small synthetic scene (every setting is accepted by the model
    and the training loop; the patch sides of all datasets)."""
    import numpy as np
    import torch
    import symbionet.data as D
    from symbionet.runs import run_one, write_config

    rng = np.random.default_rng(0)
    H, W, K = 16, 18, 3
    labels = rng.integers(-1, K, (H, W))
    labels[0, :K] = np.arange(K)
    scene = D.Scene('synthetic', rng.random((6, H, W)).astype(np.float32), rng.random((1, H, W)).astype(np.float32),
                    labels, np.ones((H, W), bool), ['c%d' % k for k in range(K)])
    split = dict(D.make_split(scene.labels, 0.3, 0), path='synthetic')
    plan = F.Plan()
    cases = [('houston2013', v) for v in plan.raw['versions']] + [(d, v) for d in ('muufl', 'trento')
                                                                  for v in ('full_patch_a', 'full_patch_b', 'full_patch_c')]
    out = Path(tempfile.mkdtemp(prefix='formal_all_'))
    try:
        for d, v in cases:
            cfg = apply(plan.config(d, v), [('training.epochs', 1), ('eval.every', 1), ('training.batch_size', 16),
                                             ('training.pool_batch', 16)])

            class St:
                pass
            st = St()
            st.dataset, st.scene, st.split = 'synthetic', scene, split
            st.pool = D.pool_indices(scene, split, size=cfg['data']['pool_size'])
            st.sampler = D.PatchSampler(scene, cfg['data']['patch'], torch.device('cpu'), 'zero')
            write_config(out, cfg)
            res = run_one(cfg, st, 0, 0, torch.device('cpu'), out, v, log=lambda *_: None)
            assert 0 <= res['best']['OA'] <= 100, (d, v)
            if cfg['data']['train_fraction'] < 1:
                assert res['n_train'] < len(split['train']), (d, v)
    finally:
        shutil.rmtree(out)


def test_finished_runs_and_estimates():
    tmp, plan = fake_root([('houston2013', 'full', 0, 94.0), ('houston2013', 'full', 1, 95.0),
                           ('muufl', 'base', 2, 90.0)])
    try:
        assert plan.done('houston2013', 'full', 0)['best']['OA'] == 94.0
        assert plan.done('houston2013', 'full', 2) is None and plan.done('trento', 'full', 0) is None
        (plan.folder('muufl', 'base') / 'ckpt' / 'run2_last.pt').unlink()      # a run without its weights: not done
        assert plan.done('muufl', 'base', 2) is None
        m = F.measured_minutes(plan, 'stage1', 'cpu')
        assert m[('houston2013', 'full')] == [9.0, 9.0] and m[('houston2013', 'm2c')] == [9.0, 9.0]
        assert F.minutes(plan, 'houston2013', 'full', m) == 9.0
        assert F.minutes(plan, 'houston2013', 'copies_m2c', m) == 9.0          # the same kind on the same dataset
        assert F.minutes(plan, 'muufl', 'copies', m) == 7.5                    # nothing measured: the table
        assert F.measured_minutes(plan, 'stage1', 'RTX') == {}                 # runs of another machine do not count
        text = F.summary(plan, 'stage1')
        assert ' 2/5 ' in text and '94.50 ± 0.50' in text and '0:94.00 1:95.00' in text
        assert '2 of 135 runs finished' in text
        assert 'check against the screening runs, houston2013 full: 0:94.00 1:95.00 here, 0:94.06 1:94.60 there '\
               '-> DIFFERENT' in text
    finally:
        shutil.rmtree(tmp)


def test_same_settings_and_code_id():
    plan = F.Plan()
    a, b = plan.config('trento', 'full'), plan.config('trento', 'copies_m2c')
    assert F.same_settings(a, a) == [] and F.same_settings(a, b) == ['model.pre_gate', 'model.pre_gate_p']
    assert F.same_settings(a, dict(a, runs=3, seed=7, print_every=1)) == []
    tmp = Path(tempfile.mkdtemp(prefix='formal_code_'))
    try:
        for f in F.CODE:
            (tmp / f).parent.mkdir(parents=True, exist_ok=True)
            (tmp / f).write_bytes((ROOT / f).read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        assert F.code_id(tmp) == F.code_id(ROOT), 'the code id must not depend on the line endings'
        assert len(F.code_id(ROOT)) == len(F.CODE)
    finally:
        shutil.rmtree(tmp)


def test_run_one_saves_the_best_and_the_last_epoch():
    """A run of the full method on a small synthetic scene: both sets of weights are on disk and give back the
    scores that the record holds."""
    import numpy as np
    import torch
    import symbionet.data as D
    from symbionet.engine import build_model, evaluate
    from symbionet.runs import read_run, run_one, write_config, write_summary

    rng = np.random.default_rng(0)
    H, W, K = 16, 18, 3
    labels = rng.integers(-1, K, (H, W))
    labels[0, :K] = np.arange(K)
    scene = D.Scene('synthetic', rng.random((6, H, W)).astype(np.float32), rng.random((1, H, W)).astype(np.float32),
                    labels, np.ones((H, W), bool), ['c%d' % k for k in range(K)])
    plan = F.Plan()
    cfg = apply(plan.config('trento', 'full'), [('data.patch', 5), ('training.epochs', 3), ('eval.every', 1),
                                                 ('training.batch_size', 16), ('training.pool_batch', 16)])

    class St:                                            # what runs.Setting holds, for a scene that is not on disk
        pass
    st = St()
    st.dataset, st.scene = 'synthetic', scene
    st.split = dict(D.make_split(scene.labels, 0.3, 0), path='synthetic')
    st.pool = D.pool_indices(scene, st.split)
    st.sampler = D.PatchSampler(scene, 5, torch.device('cpu'), 'zero')
    out = Path(tempfile.mkdtemp(prefix='formal_run_'))
    try:
        write_config(out, cfg)
        res = run_one(cfg, st, 4, 4, torch.device('cpu'), out, 'full', extra=dict(code={'x': 'y'}), log=lambda *_: None)
        back = read_run(out, 4)
        assert back is not None and back['seed'] == 4 and back['code'] == {'x': 'y'} and back['run'] == 4
        assert read_run(out, 0) is None and not list(out.glob('*.tmp'))
        assert len(res['history']) == 3 and 'state' not in back and 'last_state' not in back
        test = torch.from_numpy(st.split['test'])
        for which in ('best', 'last'):
            model = build_model(cfg, scene)
            model.load_state_dict(torch.load(out / 'ckpt' / ('run4_%s.pt' % which)))
            got = evaluate(model, st.sampler, test, scene.n_classes, 64)
            assert abs(got['OA'] - back[which]['OA']) < 1e-9, (which, got['OA'], back[which]['OA'])
        assert write_summary(out, [back], scene.classes)['runs'] == 1 and (out / 'summary.json').exists()
    finally:
        shutil.rmtree(out)


def test_smoke_plan_is_separate():
    plan = F.Plan(smoke=True)
    c = plan.config('muufl', 'full')
    assert plan.exp == 'formal_smoke' and c['training']['epochs'] == 2 and c['eval']['subset'] == 300
    assert F.hm(59.6) == '1 h 00 min' and F.hm(9.2) == '9 min' and F.hm(1039) == '17 h 19 min'


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
    print('test_formal: %d/%d passed' % (len(tests) - len(failed), len(tests)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
