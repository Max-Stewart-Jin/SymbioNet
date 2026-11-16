"""Tests of symbionet/external.py (published networks in the pipeline of this project) and of the waiting rule of
compare.py. Run with `python tests/test_external.py` (prints a summary) or with pytest.

The first group needs nothing but this project. The second group builds the four published networks and is
skipped when their sources are not there: put the downloaded repositories under external/ (see the top of
symbionet/external.py), or name their folder in the environment variable SYMBIONET_EXTERNAL.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare as C  # noqa: E402
import formal as F  # noqa: E402
import symbionet.data as D  # noqa: E402
import symbionet.testing as T  # noqa: E402
from symbionet import external as X, mamba_compat  # noqa: E402
from symbionet.config import apply  # noqa: E402
from symbionet.corrupt import Condition  # noqa: E402
from symbionet.engine import evaluate  # noqa: E402

CPU = torch.device('cpu')


def quiet(*_):
    pass


def synthetic(bands=30, patch=9, classes=4, size=(22, 26), epochs=2, seed=0):
    """A small random scene with its split, patch sampler and the settings of a short run of the protocol."""
    rng = np.random.default_rng(seed)
    H, W = size
    labels = rng.integers(-1, classes, (H, W))
    labels[0, :classes] = np.arange(classes)
    labels[1, :classes] = np.arange(classes)
    scene = D.Scene('synthetic', rng.random((bands, H, W)).astype(np.float32),
                    rng.random((1, H, W)).astype(np.float32), labels, np.ones((H, W), bool),
                    ['c%d' % k for k in range(classes)])
    split = dict(D.make_split(scene.labels, 0.3, 0), path='synthetic.npz')
    cfg = apply(F.Plan().config('trento', C.PROTOCOL), [('data.patch', patch), ('training.epochs', epochs),
                                                         ('eval.every', 1), ('training.batch_size', 16)])
    return scene, split, D.PatchSampler(scene, patch, CPU, 'zero'), cfg


def published():
    """The folder with the downloaded repositories, or None."""
    for place in (os.environ.get('SYMBIONET_EXTERNAL'), ROOT / 'external'):
        if place and Path(place).is_dir():
            try:
                for m in X.SOURCES:
                    if m in X.METHODS:
                        X.locate(m, place)
                return Path(place)
            except FileNotFoundError:
                continue
    return None


class without:
    """Inside the block, the named packages cannot be imported (as on a machine where they are not installed)."""

    def __init__(self, *names):
        self.names, self.saved = names, {}

    def __enter__(self):
        for n in self.names:
            self.saved[n] = {k: v for k, v in sys.modules.items() if k == n or k.startswith(n + '.')}
            for k in self.saved[n]:
                del sys.modules[k]
            sys.modules[n] = None

    def __exit__(self, *exc):
        for n in self.names:
            sys.modules.pop(n, None)
            sys.modules.update(self.saved[n])


# ------------------------------------------------------------------------------------- nothing but this project
def test_definitions_runs_only_the_definitions():
    source = '\n'.join(['import nonexistent_package_xyz', 'raise RuntimeError("the script was run")',
                        'DATA = open("/nonexistent/file").read()',
                        'class Net(nn.Module):', '    def __init__(self):', '        super().__init__()',
                        '        self.fc = nn.Linear(NC, 2)', '    def forward(self, x):', '        return self.fc(x)',
                        'def helper(x):', '    return x + 1',
                        'for epoch in range(10 ** 9):', '    pass'])
    space = X.definitions(source, dict(nn=nn, NC=3))
    assert space['helper'](1) == 2 and 'DATA' not in space
    assert space['Net']()(torch.zeros(4, 3)).shape == (4, 2)


def test_notebook_cell():
    tmp = Path(tempfile.mkdtemp(prefix='external_'))
    try:
        nb = dict(cells=[dict(cell_type='markdown', source=['# THE MODEL']),
                         dict(cell_type='code', source=['x = 1\n', 'print(x)']),
                         dict(cell_type='code', source=['# THE MODEL\n', '!pip install something\n',
                                                        '%matplotlib inline\n', 'class A:\n', '    pass\n']),
                         dict(cell_type='code', source=['# THE MODEL again\n', 'class B:\n', '    pass'])])
        (tmp / 'n.ipynb').write_text(json.dumps(nb), encoding='utf-8')
        text = X.notebook_cell(tmp / 'n.ipynb', 'THE MODEL')
        assert 'class A' in text and 'class B' not in text and 'pip' not in text and 'matplotlib' not in text
        try:
            X.notebook_cell(tmp / 'n.ipynb', 'NOT THERE')
            raise AssertionError('a missing cell must be reported')
        except ValueError:
            pass
    finally:
        shutil.rmtree(tmp)


def test_locate_and_hash():
    tmp = Path(tempfile.mkdtemp(prefix='external_'))
    saved = dict(X.SOURCES['endnet'])
    try:
        try:
            X.locate('endnet', tmp)
            raise AssertionError('a missing source must be reported')
        except FileNotFoundError as e:
            assert 'EndNet' in str(e) and 'github.com' in str(e)
        a = tmp / 'Some-Folder-main' / 'model' / 'EndNet.py'
        b = tmp / 'older copy' / 'deep' / 'model' / 'EndNet.py'
        other = tmp / 'Some-Folder-main' / 'notmodel' / 'EndNet.py'           # the name alone is not enough
        for f, text in ((a, b'class EndNet:\r\n    pass\r\n'), (b, b'class EndNet:\n    old = 1\n'),
                        (other, b'class EndNet:\n    pass\n')):
            f.parent.mkdir(parents=True)
            f.write_bytes(text)
        assert X.source_hash(a) == X.source_hash(other)                       # line endings are ignored
        X.SOURCES['endnet']['sha256'] = X.source_hash(a)
        path, warning = X.locate('endnet', tmp)
        assert path == a and warning == ''
        X.SOURCES['endnet']['sha256'] = X.source_hash(b)                      # the copy with the tested hash wins
        assert X.locate('endnet', tmp) == (b, '')
        X.SOURCES['endnet']['sha256'] = '0' * 64                              # no copy is the tested one
        path, warning = X.locate('endnet', tmp)
        assert path in (a, b) and 'not the version' in warning
    finally:
        X.SOURCES['endnet'] = saved
        shutil.rmtree(tmp)


def test_pca_basis():
    rng = np.random.default_rng(3)
    C_, H, W, k = 12, 9, 11, 5
    hsi = (rng.random((C_, H, W)) * np.linspace(1, 4, C_)[:, None, None]).astype(np.float32)
    valid = rng.random((H, W)) > 0.2
    mean, comps = X.pca_basis(hsi, valid, k)
    x = hsi[:, valid].astype(np.float64).T                                    # (n, C)
    assert np.allclose(mean, x.mean(0)) and comps.shape == (k, C_)
    assert np.allclose(comps @ comps.T, np.eye(k), atol=1e-10)                # orthonormal
    z = (x - mean) @ comps.T
    var = z.var(0, ddof=1)
    assert np.all(np.diff(var) <= 1e-12)                                      # ordered by variance
    s = np.linalg.svd(x - mean, compute_uv=False)                             # the largest variances there are
    assert np.allclose(var, s[:k] ** 2 / (len(x) - 1))
    assert np.allclose(np.cov(z.T), np.diag(var), atol=1e-10)                 # uncorrelated
    assert all(c[np.abs(c).argmax()] > 0 for c in comps)                      # the sign is fixed
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return
    ref = PCA(n_components=k).fit(x)
    flip = np.sign((ref.components_ * comps).sum(1))                          # sklearn fixes the sign another way
    assert np.allclose(ref.components_ * flip[:, None], comps, atol=1e-6)
    assert np.allclose(ref.transform(x) * flip, z, atol=1e-6)


def test_rearrange_without_einops():
    x = torch.arange(24.0).reshape(2, 3, 4)
    with without('einops'):
        assert torch.equal(X.rearrange(x, 'b h w -> b w h'), x.transpose(1, 2))
        try:
            X.rearrange(x, 'b h w -> b (h w)')
            raise AssertionError('an unknown pattern must be reported')
        except NotImplementedError:
            pass
    if X._importable('einops'):                                               # and with einops, the same
        assert torch.equal(X.rearrange(x, 'b h w -> b w h'), x.transpose(1, 2))


def test_stand_ins_for_missing_packages():
    before = {k: sys.modules.get(k) for k in ('mamba_ssm', 'thop')}
    with without('mamba_ssm', 'thop'):
        with X.stand_ins() as used:
            from mamba_ssm import Mamba
            import thop
            assert Mamba is mamba_compat.Mamba and used['mamba'].startswith('symbionet.mamba_compat')
            try:
                thop.profile()
                raise AssertionError('the stand-in of thop must not pretend to count')
            except RuntimeError:
                pass
        assert 'mamba_ssm' not in sys.modules and 'thop' not in sys.modules   # removed again
    assert {k: sys.modules.get(k) for k in before} == before


def test_take_lock_waits_for_its_turn():
    tmp = Path(tempfile.mkdtemp(prefix='compare_lock_'))
    plan = types.SimpleNamespace(out_root=tmp, smoke=False)
    held = []

    def timed(**kw):
        t0 = time.time()
        lock = C.take_lock(plan, **kw)
        held.append(lock)
        return lock, time.time() - t0

    try:
        lock, dt = timed(wait=True, quiet=5, poll=0.1)                        # nothing there: at once
        assert lock and dt < 1
        assert C.take_lock(plan, False) is None                               # taken: no second one
        lock.close()
        (tmp / 'log.txt').write_text('a line of the plan')                    # the plan worked a moment ago
        lock, dt = timed(wait=True, quiet=1.5, poll=0.1)
        assert lock and 1.2 < dt < 4, dt
        lock.close()
        (tmp / 'log.txt').write_text('another line')
        lock, dt = timed(wait=False)                                          # --no-wait: no quiet time
        assert lock and dt < 1
        lock.close()
        package = F.single_instance(tmp / 'launcher.lock')                    # a package is working
        assert package and C.take_lock(plan, False) is None
        threading.Timer(1.0, package.close).start()
        lock, dt = timed(wait=True, quiet=1.0, poll=0.1)                      # waits for it, then for the quiet time
        assert lock and 1.8 < dt < 5, dt
        lock.close()
        (tmp / 'log.txt').write_text('a third line')
        plan.smoke = True                                                     # try-out runs have no queue
        lock, dt = timed(wait=True, quiet=30, poll=0.1)
        assert lock and dt < 1
    finally:
        for f in held:
            if f:
                f.close()
        shutil.rmtree(tmp)


class Toy(X.Compared):
    """The smallest compared network: a linear classifier of the band means of a patch."""

    def __init__(self, src, scene, patch):
        super().__init__()
        self.net = nn.Linear(scene.hsi.shape[0] + scene.lidar.shape[0], scene.n_classes)
        self.calls, self.seen = 0, 0

    def forward(self, h, l, on=None):
        return self.net(torch.cat([h.mean((2, 3)), l.mean((2, 3))], 1))

    def before_eval(self, batches):
        self.calls += 1
        self.seen = sum(len(y) for _, _, y in batches())


def test_training_of_a_compared_network():
    tmp = Path(tempfile.mkdtemp(prefix='external_'))
    built = []
    try:
        src = tmp / 'Toy-main' / 'toy' / 'toy.py'
        src.parent.mkdir(parents=True)
        src.write_text('# nothing to read\n')
        X.SOURCES['toy'] = dict(name='Toy', file='toy/toy.py', repo='-', sha256=X.source_hash(src))
        X.METHODS['toy'] = lambda *a: built.append(Toy(*a)) or built[-1]
        scene, split, sampler, cfg = synthetic(bands=6, patch=5, classes=3, epochs=4)
        rec = X.train(cfg, 'toy', scene, sampler, split, 7, CPU, tmp, log=quiet)
        assert {'seed', 'best_epoch', 'best', 'last', 'history', 'seconds', 'parameters', 'compared', 'state',
                'last_state'} <= set(rec) and rec['seed'] == 7
        assert [h['epoch'] for h in rec['history']] == [1, 2, 3, 4]
        assert all(set(h) == {'epoch', 'test_oa', 'loss'} and np.isfinite(h['loss']) for h in rec['history'])
        oas = [h['test_oa'] for h in rec['history']]
        # the curve is kept in single precision (as in engine.train_run), the scores in double precision
        assert abs(rec['best']['OA'] - max(oas)) < 1e-4 and rec['best_epoch'] == 1 + oas.index(max(oas))
        assert abs(rec['last']['OA'] - oas[-1]) < 1e-4
        assert built[0].calls == 4 and built[0].seen == len(split['train'])   # before every scoring, all training patches
        assert rec['parameters'] == dict(test_time=7 * 3 + 3, training_only=0)
        info = rec['compared']
        assert info['method'] == 'toy' and info['as_tested'] and 'warning' not in info and info['source'] == str(src)
        idx = torch.from_numpy(split['test'])
        for which, state in (('best', rec['state']), ('last', rec['last_state'])):   # the weights give the scores
            model = X.build('toy', scene, cfg['data']['patch'], tmp)
            model.load_state_dict(state)
            assert abs(evaluate(model, sampler, idx, scene.n_classes, 64)['OA'] - rec[which]['OA']) < 1e-9
        again = X.train(cfg, 'toy', scene, sampler, split, 7, CPU, tmp, log=quiet)   # the same seed, the same run
        assert again['history'] == rec['history']
        other = X.train(cfg, 'toy', scene, sampler, split, 8, CPU, tmp, log=quiet)
        assert other['history'] != rec['history']
        src.write_text('# another version\n')
        info = X.train(cfg, 'toy', scene, sampler, split, 7, CPU, tmp, log=quiet)['compared']
        assert not info['as_tested'] and 'not the version' in info['warning']
        try:
            X.build('nothing', scene, 5, tmp)
            raise AssertionError('an unknown method must be reported')
        except ValueError:
            pass
    finally:
        X.SOURCES.pop('toy', None)
        X.METHODS.pop('toy', None)
        shutil.rmtree(tmp)


# ----------------------------------------------------------------------------------- with the published sources
def test_published_networks_take_a_training_step():
    ext = published()
    if ext is None:
        return 'skipped: the published sources are not under external/'
    scene, split, sampler, cfg = synthetic()
    idx = torch.from_numpy(split['train'][:12])
    h, l = sampler(idx)
    y = sampler.targets(idx)
    ce = nn.CrossEntropyLoss(label_smoothing=0.2)
    for method in ('dahgm', 'cnnh', 'endnet', 'mft'):
        torch.manual_seed(0)
        model = X.build(method, scene, cfg['data']['patch'], ext)
        assert model.info['method'] == method and model.cfg == {'mask_labeled': False} and model.branch is False
        assert model.strip_training_branch() is model
        before = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        loss = model.training_loss(h, l, y, ce)
        assert loss.ndim == 0 and torch.isfinite(loss), method
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), method
        assert all(g is None or torch.isfinite(g).all() for g in grads), method
        opt.step()
        assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items()), method
        model.before_eval(lambda: [(h, l, y)])
        model.eval()
        with torch.no_grad():
            out, again = model(h, l), model(h, l)
        assert out.shape == (12, scene.n_classes) and torch.isfinite(out).all(), method
        assert torch.equal(out, again), method                               # nothing random at test time
        n = model.count_parameters()
        assert n['test_time'] > 0 and n['test_time'] + n['training_only'] == sum(
            p.numel() for p in model.parameters() if p.requires_grad), method
        twin = X.build(method, scene, cfg['data']['patch'], ext)             # the weights carry everything
        twin.load_state_dict(model.state_dict())
        twin.eval()
        with torch.no_grad():
            assert torch.allclose(twin(h, l), out, atol=1e-6), method


def test_published_networks_without_optional_packages():
    ext = published()
    if ext is None:
        return 'skipped: the published sources are not under external/'
    scene, split, sampler, cfg = synthetic()
    idx = torch.from_numpy(split['train'][:6])
    h, l = sampler(idx)
    real_mamba = X._importable('mamba_ssm')
    outs = {}
    for blocked in ((), ('einops', 'thop', 'mamba_ssm', 'sklearn', 'matplotlib', 'torchvision', 'scipy')):
        with without(*blocked):
            for method in ('dahgm', 'cnnh', 'endnet', 'mft'):
                torch.manual_seed(0)
                model = X.build(method, scene, cfg['data']['patch'], ext).eval()
                if method == 'dahgm' and 'mamba_ssm' in blocked:
                    assert model.info['mamba'].startswith('symbionet.mamba_compat')
                    assert sum(isinstance(m, mamba_compat.Mamba) for m in model.modules()) == 2
                with torch.no_grad():
                    outs.setdefault(method, []).append(model(h, l))
    for method, (a, b) in outs.items():                                      # the same network either way
        if method == 'dahgm' and real_mamba:                                 # the kernel of mamba_ssm the first time
            continue
        assert torch.isfinite(a).all() and torch.equal(a, b), method


def test_coupled_cnn_preprocessing_and_decision_rule():
    ext = published()
    if ext is None:
        return 'skipped: the published sources are not under external/'
    scene, split, sampler, cfg = synthetic()
    torch.manual_seed(0)
    model = X.build('cnnh', scene, cfg['data']['patch'], ext).eval()
    idx = torch.from_numpy(split['train'])
    h, l = sampler(idx)
    y = sampler.targets(idx)
    mean, comps = X.pca_basis(scene.hsi, scene.valid, 20)                    # the 20 principal components of a patch
    want = torch.einsum('bcij,kc->bkij', h.double() - torch.tensor(mean)[None, :, None, None], torch.tensor(comps))
    got = torch.nn.functional.conv2d(h, model.pca_w, model.pca_b)
    assert got.shape == (len(idx), 20, 9, 9) and torch.allclose(got.double(), want, atol=1e-4)
    assert torch.equal(model.w, torch.tensor([[0.0] * 4, [0.0] * 4, [1.0] * 4]))   # before training: the fused output
    with torch.no_grad():
        o1, o2, o3 = model.heads(h, l)
        assert torch.equal(model(h, l), o3)
        model.before_eval(lambda: [(h[:40], l[:40], y[:40]), (h[40:], l[40:], y[40:])])
        right = torch.stack([torch.bincount(y[o.argmax(1) == y], minlength=4).float() for o in (o1, o2, o3)])
        assert torch.allclose(model.w, right / (right.sum(0, keepdim=True) + 1e-5))
        assert torch.allclose(model(h, l), model.w[2] * o3 + model.w[1] * o2 + model.w[0] * o1)
    ce = nn.CrossEntropyLoss()
    model.train()
    torch.manual_seed(1)
    a = model.training_loss(h[:16], l[:16], y[:16], ce)
    torch.manual_seed(1)
    p1, p2, p3 = model.heads(h[:16], l[:16])
    assert torch.allclose(a, 0.01 * ce(p1, y[:16]) + 0.01 * ce(p2, y[:16]) + ce(p3, y[:16]))
    assert {'w', 'pca_w', 'pca_b'} <= set(model.state_dict())               # kept with the weights


def test_endnet_reads_the_centre_pixel():
    ext = published()
    if ext is None:
        return 'skipped: the published sources are not under external/'
    scene, split, sampler, cfg = synthetic()
    torch.manual_seed(0)
    model = X.build('endnet', scene, cfg['data']['patch'], ext).eval()
    idx = torch.from_numpy(split['train'][:8])
    h, l = sampler(idx)
    h2, l2 = torch.rand_like(h), torch.rand_like(l)
    h2[:, :, 4, 4], l2[:, :, 4, 4] = h[:, :, 4, 4], l[:, :, 4, 4]
    with torch.no_grad():
        assert torch.equal(model(h, l), model(h2, l2))                       # only the centre counts
        h2[:, :, 4, 4] += 0.5
        assert not torch.equal(model(h, l), model(h2, l2))
    n = model.count_parameters()
    assert n['training_only'] == sum(p.numel() for k, p in model.net.named_parameters() if k.startswith('decoder')) > 0


def test_a_published_network_is_trained_and_scored_like_a_version_of_the_plan():
    ext = published()
    if ext is None:
        return 'skipped: the published sources are not under external/'
    tmp = Path(tempfile.mkdtemp(prefix='external_'))
    try:
        scene, split, sampler, cfg = synthetic(epochs=3)
        members = []
        for method in ('cnnh', 'endnet'):
            rec = X.train(cfg, method, scene, sampler, split, 0, CPU, ext, log=quiet)
            assert rec['compared']['as_tested'], rec['compared'].get('warning')
            ckpt = tmp / ('%s_best.pt' % method)
            torch.save(rec['state'], ckpt)
            members.append(T.Member('%s/best' % C.version(method), 0, cfg, ckpt, rec['best']['OA']))
        bench = T.Bench(scene, split, cfg['data'], CPU, chunk=64, block=4)
        for m in members:                                                    # as compare.py test does
            assert bench.fits(m) == []
            model = X.build(m.key.split('/')[0][len('ext_'):], bench.scene, m.cfg['data']['patch'], ext)
            model.load_state_dict(torch.load(m.ckpt, map_location='cpu', weights_only=True))
            m.model = model.eval()
        for m, s in zip(members, bench.score(members, None)):                # the clean scores of the training record
            assert abs(s['OA'] - m.recorded) < 1e-9, (m.key, s['OA'], m.recorded)
        for cond in (Condition('missing', 'lidar', 1.0), Condition('block', 'hsi', 0.3),
                     Condition('bands', 'hsi', 0.5), Condition('noise', 'both', 10.0)):
            for s in bench.score(members, cond):
                assert 0 <= s['OA'] <= 100
        first = bench.score(members, Condition('pixel', 'both', 0.5))
        assert first == bench.score(members, Condition('pixel', 'both', 0.5))   # the same test scene every time
    finally:
        shutil.rmtree(tmp)


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    failed, skipped = [], 0
    for name, fn in tests:
        try:
            note = fn()
            skipped += bool(note)
            print('SKIP' if note else 'PASS', name, '- ' + note if note else '')
        except Exception as e:                  # noqa: BLE001 - report every failure, keep going
            failed.append(name)
            print('FAIL', name, '-', type(e).__name__, e)
    print('=' * 60)
    print('test_external: %d/%d passed%s' % (len(tests) - len(failed) - skipped, len(tests),
                                             ', %d skipped (no published sources)' % skipped if skipped else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
