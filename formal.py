"""The experiments of the paper (configs/formal.yaml).

  python formal.py plan [stage]       what is planned, what is on disk, how long the rest takes
  python formal.py run <stage>        train what is missing of a training stage, then print the summary
  python formal.py test <stage>       score what is missing of a test stage, then print the summary
  python formal.py summary [stage]    the summary of what is on disk
  python formal.py timing             seconds per training epoch and milliseconds per test patch on this machine
  python formal.py maps               the predicted class of every pixel, for the classification maps of the paper
  python formal.py pack               results/SymbioNet_formal_results.zip: every record, without the weights

A run is one seed of one version on one dataset, saved as results/formal/<dataset>/<version>/run<seed>.json with
the weights of the best and of the last epoch (symbionet/runs.py). Finished runs are never trained again, so
`run` can be stopped and started again at any time; a run that was interrupted starts over.

A test stage scores finished runs under test conditions without training (symbionet/testing.py): the corruptions
of the paper, a shifted LiDAR raster, another way of filling lost values, and the accuracy inside the cloud shadow
of Houston2013. The scores of a dataset are kept in results/formal/tests/<dataset>.json and are written after every
condition, so `test` can also be stopped and started again at any time.

Everything printed by `run` and `test` is also appended to results/formal/log.txt.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # deterministic cuBLAS; set before CUDA starts

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from symbionet.conditions import Refill, Shift, apply as apply_condition, n_draws, suite  # noqa: E402
from symbionet.data import PatchSampler  # noqa: E402
from symbionet.config import apply, dataset_config, load_yaml  # noqa: E402
from symbionet.engine import build_model, predict, train_run  # noqa: E402
from symbionet.runs import Setting, find_data_root, mean_std, read_run, run_one, write_config, write_summary  # noqa: E402
from symbionet.testing import CLEAN, Bench, Member, Store, label_of, load_zones, mean_over_seeds, zone_scores  # noqa: E402

PLAN = ROOT / 'configs' / 'formal.yaml'
# the files that decide what a training run computes; their hashes are stored with every run
CODE = ('symbionet/model.py', 'symbionet/engine.py', 'symbionet/data.py', 'symbionet/corrupt.py',
        'symbionet/runs.py', 'symbionet/metrics.py', 'configs/default.yaml', 'configs/datasets.yaml')
SMOKE = {'training.epochs': 2, 'eval.every': 1, 'eval.subset': 300, 'print_every': 1}    # --smoke: checks the setup
NOT_COMPARED = ('runs', 'seed', 'print_every')        # settings that do not change what a run computes


# ------------------------------------------------------------------------------------------------------ plan
class Plan:
    def __init__(self, path=PLAN, root=ROOT, smoke=False):
        self.root, self.raw, self.smoke = Path(root), load_yaml(path), smoke
        self.exp = self.raw['exp'] + ('_smoke' if smoke else '')
        self.seeds = list(self.raw['seeds'])
        self.out_root = self.root / 'results' / self.exp
        self._cfg = {}

    def stage(self, name: str) -> dict:
        if name not in self.raw['stages']:
            raise SystemExit('unknown stage %r (stages: %s)' % (name, ', '.join(self.raw['stages'])))
        return self.raw['stages'][name]

    def overrides(self, dataset: str, version: str) -> list:
        """The (key, value) pairs put on top of the dataset's default settings, in the order they are applied."""
        if version not in self.raw['versions']:
            raise SystemExit('unknown version %r' % version)
        if dataset not in self.raw['datasets']:
            raise SystemExit('dataset %r has no entry in the plan' % dataset)
        pairs = list(self.raw.get('common', {}).items()) + list(self.raw['datasets'][dataset].items())
        patch = dict(pairs).get('data.patch')

        def value(v):
            """$patch: the patch side of the dataset; $name: the entry of the dataset in the table `name`."""
            if v == '$patch':
                return patch
            if isinstance(v, str) and v.startswith('$'):
                table = (self.raw.get('values') or {}).get(v[1:])
                if not table or dataset not in table:
                    raise SystemExit('configs/formal.yaml: no value %s for %s' % (v, dataset))
                return table[dataset]
            return v

        pairs += [(k, value(v)) for k, v in (self.raw['versions'][version] or {}).items()]
        if self.smoke:
            pairs += list(SMOKE.items())
        return pairs

    def config(self, dataset: str, version: str) -> dict:
        key = (dataset, version)
        if key not in self._cfg:
            self._cfg[key] = apply(dataset_config(self.root, dataset), self.overrides(dataset, version))
        return self._cfg[key]

    def kind(self, dataset: str, version: str) -> str:
        m = self.config(dataset, version)['model']
        return 'm2c' if m['decoder'] else ('copies' if m['mask_labeled'] else 'base')

    def folder(self, dataset: str, version: str) -> Path:
        return self.out_root / dataset / version

    def groups(self, stage: str) -> list:
        """[(dataset, version, seeds)] in the order of training: version by version, the datasets inside."""
        st = self.stage(stage)
        seeds = list(st.get('seeds', self.seeds))
        return [(d, v, seeds) for v in st['versions'] for d in st['datasets']]

    def runs(self, stage: str) -> list:
        return [(d, v, s) for d, v, seeds in self.groups(stage) for s in seeds]

    def done(self, dataset: str, version: str, seed: int):
        """The record of the finished run, or None."""
        res = read_run(self.folder(dataset, version), seed)
        return res if res is not None and res.get('seed') == seed else None

    # ---- test stages
    def is_test(self, name: str) -> bool:
        return name in (self.raw.get('tests') or {})

    def tests(self, name: str) -> dict:
        if not self.is_test(name):
            raise SystemExit('unknown test stage %r (test stages: %s)'
                             % (name, ', '.join(self.raw.get('tests') or {}) or 'none'))
        return self.raw['tests'][name]

    def parts(self, name: str) -> list:
        """The parts of a test stage, in the order they are done: [dict(name, suites, conds, versions, weights)]."""
        t = self.tests(name)
        st = self.stage(t['of'])
        out = []
        for p in t['parts']:
            versions = list(st['versions']) if p.get('versions', 'all') == 'all' else list(p['versions'])
            unknown = [v for v in versions if v not in st['versions']]
            if unknown:
                raise SystemExit('%s, part %s: %s not in %s' % (name, p['name'], ', '.join(unknown), t['of']))
            weights = p.get('weights', 'best')
            if weights not in ('best', 'last'):
                raise SystemExit('%s, part %s: weights must be best or last' % (name, p['name']))
            conds = [c for s in p['suites'] for c in suite(s)]
            out.append(dict(name=p['name'], suites=list(p['suites']), conds=thin(conds) if self.smoke else conds,
                            versions=versions, weights=weights))
        return out

    def store_path(self, dataset: str) -> Path:
        return self.out_root / 'tests' / ('%s.json' % dataset)

    def members(self, stage: str, part: dict, dataset: str) -> list:
        """The finished runs of a training stage that a part (its versions, its weights) covers on a dataset, with
        the settings they were trained with."""
        st = self.stage(stage)
        out = []
        for v in part['versions']:
            folder, cfg = self.folder(dataset, v), None
            for s in st.get('seeds', self.seeds):
                r = self.done(dataset, v, s)
                if r is None:
                    continue
                if cfg is None:
                    cfg = load_yaml(folder / 'config.yaml')
                out.append(Member('%s/%s' % (v, part['weights']), s, cfg,
                                  folder / 'ckpt' / ('run%d_%s.pt' % (s, part['weights'])),
                                  r[part['weights']]['OA']))
        return out


def thin(conds: list) -> list:
    """--smoke: the first and the last level of every test and target (every code path, few test scenes)."""
    groups = {}
    for c in conds:
        groups.setdefault((type(c).__name__, c.test, c.target), []).append(c)
    keep = {id(g[0]) for g in groups.values()} | {id(g[-1]) for g in groups.values()}
    return [c for c in conds if id(c) in keep]


def same_settings(a: dict, b: dict) -> list:
    """The settings in which two resolved configs differ (those that change what a run computes)."""
    diff = []
    for sec in sorted(set(a) | set(b)):
        if sec in NOT_COMPARED:
            continue
        x, y = a.get(sec), b.get(sec)
        if isinstance(x, dict) and isinstance(y, dict):
            diff += ['%s.%s' % (sec, k) for k in sorted(set(x) | set(y)) if x.get(k) != y.get(k)]
        elif x != y:
            diff.append(sec)
    return diff


def code_id(root=ROOT) -> dict:
    """{file: first 10 hex digits of its SHA-256}; line endings are ignored, so Windows and Linux copies agree."""
    out = {}
    for f in CODE:
        data = (Path(root) / f).read_bytes().replace(b'\r\n', b'\n')
        out[f] = hashlib.sha256(data).hexdigest()[:10]
    return out


# -------------------------------------------------------------------------------------------------- estimates
def minutes(plan: Plan, dataset: str, version: str, measured: dict) -> float:
    """Expected minutes of one run: the finished runs of the same version, else of the same kind, else the table."""
    kind = plan.kind(dataset, version)
    for key in ((dataset, version), (dataset, kind)):
        if measured.get(key):
            return sum(measured[key]) / len(measured[key])
    if plan.smoke:                                      # the table is for full runs
        return 1.0
    return float(plan.raw.get('minutes', {}).get(kind, {}).get(dataset, 10.0))


def measured_minutes(plan: Plan, stage: str, device_name: str) -> dict:
    """Minutes of the finished runs of this stage on this device, by (dataset, version) and (dataset, kind)."""
    out = {}
    for d, v, s in plan.runs(stage):
        res = plan.done(d, v, s)
        if res and res.get('device') == device_name:
            for key in ((d, v), (d, plan.kind(d, v))):
                out.setdefault(key, []).append(res['seconds'] / 60)
    return out


def hm(minutes_: float) -> str:
    h, m = divmod(int(round(minutes_)), 60)
    return ('%d h %02d min' % (h, m)) if h else ('%d min' % m)


# ---------------------------------------------------------------------------------------------------- output
class Tee:
    """print() that also appends to a log file."""

    def __init__(self, path=None):
        self.f = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, 'a', encoding='utf-8')

    def __call__(self, text=''):
        print(text, flush=True)
        if self.f:
            self.f.write(str(text) + '\n')
            self.f.flush()


def publish(path: Path, text: str, clipboard: bool = True) -> str:
    """Save the summary; on Windows it also goes to the clipboard, ready to be pasted. Returns where it is."""
    path.write_text(text + '\n', encoding='utf-8')
    if os.name == 'nt' and clipboard:
        try:
            cmd = "Get-Content -Raw -Encoding UTF8 -LiteralPath '%s' | Set-Clipboard" % str(path).replace("'", "''")
            subprocess.run(['powershell', '-NoProfile', '-Command', cmd], check=True, timeout=60,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return 'the summary is on the clipboard and in %s' % path
        except Exception:                # noqa: BLE001 - the clipboard is a convenience
            pass
    return 'the summary is in %s' % path


def single_instance(path: Path):
    """Hold a lock file for the life of this process, so that two `run` commands never train the same runs at the
    same time. Returns the open file, None if another process holds the lock, or False if locking is not possible
    here (then nothing is guarded)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        f = open(path, 'a+')
    except OSError:
        return False
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    except Exception:                    # noqa: BLE001
        return False
    return f


def keep_awake(on: bool):
    """Windows: the computer does not go to sleep while the runs are going (the display may still turn off)."""
    if os.name != 'nt':
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))
    except Exception:                    # noqa: BLE001
        pass


def summary(plan: Plan, stage: str) -> str:
    st = plan.stage(stage)
    lines = ['SUMMARY of %s (%s): %s' % (stage, plan.exp, st.get('title', '')),
             'OA / AA / Kappa / F1 in %, mean ± std over the finished seeds; best = the epoch with the best test OA, '
             'last = the last epoch.']
    head = '%-13s %-13s %5s  %-13s %6s %6s %6s  %-13s %6s %7s  %s' % (
        'dataset', 'version', 'runs', 'OA best', 'AA', 'Kappa', 'F1', 'OA last', 'epoch', 'min/run',
        'best OA of every seed')
    total = finished = 0
    codes = {}
    for d in st['datasets']:
        lines += ['', head, '-' * len(head)]
        for v in st['versions']:
            seeds = list(st.get('seeds', plan.seeds))
            runs = [r for r in (plan.done(d, v, s) for s in seeds) if r]
            total += len(seeds)
            finished += len(runs)
            if not runs:
                lines.append('%-13s %-13s %2d/%-2d  (nothing yet)' % (d, v, 0, len(seeds)))
                continue
            for r in runs:
                key = tuple(sorted((r.get('code') or {}).items()))
                codes[key] = codes.get(key, 0) + 1
            b = {k: mean_std([r['best'][k] for r in runs]) for k in ('OA', 'AA', 'Kappa', 'F1')}
            last = mean_std([r['last']['OA'] for r in runs])
            lines.append('%-13s %-13s %2d/%-2d  %6.2f ± %-4.2f %6.2f %6.2f %6.2f  %6.2f ± %-4.2f %6.0f %7.1f  %s' % (
                d, v, len(runs), len(seeds), b['OA'][0], b['OA'][1], b['AA'][0], b['Kappa'][0], b['F1'][0],
                last[0], last[1], sum(r['best_epoch'] for r in runs) / len(runs),
                sum(r['seconds'] for r in runs) / len(runs) / 60,
                ' '.join('%d:%.2f' % (r['seed'], r['best']['OA']) for r in runs)))
    lines += ['', '%d of %d runs finished.' % (finished, total)]
    if len(codes) > 1:
        lines.append('!! the finished runs were trained with %d different versions of the code: %s'
                     % (len(codes), ', '.join('%d runs' % n for n in codes.values())))
    # development check: the screening runs with the same settings and seeds
    for d, per_version in (plan.raw.get('screening') or {}).items():
        for v, ref in per_version.items():
            if d not in st['datasets'] or v not in st['versions'] or plan.smoke:
                continue
            got = {s: plan.done(d, v, s) for s in ref}
            got = {s: r['best']['OA'] for s, r in got.items() if r}
            if got:
                same = all('%.2f' % got[s] == '%.2f' % ref[s] for s in got)
                lines.append('check against the screening runs, %s %s: %s here, %s there -> %s' % (
                    d, v, ' '.join('%d:%.2f' % (s, got[s]) for s in sorted(got)),
                    ' '.join('%d:%.2f' % (s, ref[s]) for s in sorted(got)), 'the same' if same else 'DIFFERENT'))
    return '\n'.join(lines)


def show_plan(plan: Plan, stage: str, say, device_name: str = '') -> float:
    st = plan.stage(stage)
    measured = measured_minutes(plan, stage, device_name)
    say('%s: %s' % (stage, st.get('title', '')))
    say('%-13s %s' % ('version', '  '.join('%-13s' % d for d in st['datasets'])))
    left = 0.0
    for v in st['versions']:
        cells = []
        for d in st['datasets']:
            seeds = list(st.get('seeds', plan.seeds))
            n = sum(plan.done(d, v, s) is not None for s in seeds)
            left += (len(seeds) - n) * minutes(plan, d, v, measured)
            cells.append('%-13s' % ('%d/%d done' % (n, len(seeds))))
        say('%-13s %s' % (v, '  '.join(cells)))
    say('about %s of training left on this machine' % hm(left))
    return left


# ------------------------------------------------------------------------------------------------------- run
def run(plan: Plan, stage: str, a) -> int:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    lock = single_instance(plan.out_root / 'running.lock')
    if lock is None:
        print('another formal.py is training in %s; this one stops' % plan.out_root)
        return 2
    say = Tee(plan.out_root / 'log.txt')
    only = None if not a.seeds else {int(x) for x in a.seeds.split(',')}
    extra = dict(code=code_id(plan.root), device=device_name, torch=torch.__version__)
    say('=' * 100)
    say('%s  formal.py run %s%s | %s | torch %s | %s' % (
        time.strftime('%Y-%m-%d %H:%M'), stage, ' --smoke' if plan.smoke else '', device_name, torch.__version__,
        plan.out_root))
    show_plan(plan, stage, say, device_name)

    pending = [r for r in plan.runs(stage) if plan.done(*r) is None and (only is None or r[2] in only)]
    if a.limit:
        pending = pending[:a.limit]
    if not pending:
        say('nothing to train: every run of %s is on disk' % stage)
    measured = measured_minutes(plan, stage, device_name)
    scenes, failed, trained, start = {}, {}, 0, time.time()
    st = group = None                                   # the (dataset, version) whose data are loaded
    keep_awake(True)
    try:
        for i, (d, v, s) in enumerate(pending):
            if (d, v) in failed:
                continue
            try:
                if group != (d, v):                     # a new (dataset, version): check its folder, load its data
                    st = group = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    cfg, out = plan.config(d, v), plan.folder(d, v)
                    if (out / 'config.yaml').exists() and any(plan.done(d, v, x) for x in plan.seeds):
                        diff = same_settings(load_yaml(out / 'config.yaml'), cfg)
                        if diff:
                            raise RuntimeError('the runs on disk were trained with other settings (%s); move %s '
                                               'away to train this version again' % (', '.join(diff), out))
                    st = Setting(cfg, d, plan.root, find_data_root(cfg, plan.root, a.data_root), device, scenes)
                    group = (d, v)
                    write_config(out, cfg)
                    say('')
                    say('== %s | %s | %s' % (d, v, st.scene.summary()))
                    say('   ' + st.describe())
                left = sum(minutes(plan, dd, vv, measured) for dd, vv, _ in pending[i:] if (dd, vv) not in failed)
                say('[%3d/%d] %s %s seed %d | %s | this run about %s, the rest of %s about %s (until about %s)'
                    % (i + 1, len(pending), d, v, s, time.strftime('%m-%d %H:%M'), hm(minutes(plan, d, v, measured)),
                       stage, hm(left), time.strftime('%m-%d %H:%M', time.localtime(time.time() + 60 * left))))
                res = run_one(cfg, st, s, s, device, out, v, log=say,
                              extra=dict(extra, finished=time.strftime('%Y-%m-%d %H:%M')))
                trained += 1
                ref = ((plan.raw.get('screening') or {}).get(d, {}).get(v, {}) or {}).get(s)
                if ref is not None and not plan.smoke:  # development check: the screening run with this seed
                    say('   check: the screening run with these settings and this seed had %.2f -> %s'
                        % (ref, 'the same' if '%.2f' % ref == '%.2f' % res['best']['OA'] else 'DIFFERENT'))
                for key in ((d, v), (d, plan.kind(d, v))):
                    measured.setdefault(key, []).append(res['seconds'] / 60)
                write_summary(out, [r for r in (plan.done(d, v, x) for x in plan.seeds) if r], st.scene.classes)
            except KeyboardInterrupt:
                raise
            except Exception as e:           # noqa: BLE001 - a version that fails must not stop the other versions
                failed[(d, v)] = '%s: %s' % (type(e).__name__, e)
                say('!! %s %s failed (%s); going on with the next version' % (d, v, failed[(d, v)]))
                say(traceback.format_exc())
                st = group = None
    except KeyboardInterrupt:
        say('')
        say('stopped by the user: the finished runs are on disk, the interrupted run starts over next time')
    finally:
        keep_awake(False)
    say('')
    say('trained %d runs in %s' % (trained, hm((time.time() - start) / 60)))
    text = summary(plan, stage)
    if failed:
        text += '\n' + '\n'.join('!! FAILED: %s %s (%s)' % (d, v, why) for (d, v), why in failed.items())
    say('=' * 100)
    say(text)
    say('=' * 100)
    say('(%s)' % publish(plan.out_root / ('summary_%s.txt' % stage), text, clipboard=not plan.smoke))
    if lock:
        lock.close()
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ test stages
def _jobs(plan: Plan, name: str) -> list:
    """What a test stage still has to score: [(part, dataset, members, [(condition, members without a score)])]."""
    st = plan.stage(plan.tests(name)['of'])
    jobs = []
    for part in plan.parts(name):
        for d in st['datasets']:
            ms = plan.members(plan.tests(name)['of'], part, d)
            store = Store.read(plan.store_path(d))
            work = []
            for cond in [None] + part['conds']:
                missing = [m for m in ms if store is None or not store.has(m.key, label_of(cond), m.seed)]
                if missing:
                    work.append((cond, missing))
            jobs.append((part, d, ms, work))
    return jobs


def _evals(work: list) -> int:
    """Model evaluations (one model on all test pixels) in a list of (condition, members)."""
    return sum((1 if cond is None else n_draws(cond)) * len(ms) for cond, ms in work)


def _planned(plan: Plan, name: str) -> tuple:
    """(scores on disk, scores planned) of a test stage, counted per condition and run."""
    st = plan.stage(plan.tests(name)['of'])
    seeds = list(st.get('seeds', plan.seeds))
    done = total = 0
    for part in plan.parts(name):
        for d in st['datasets']:
            store = Store.read(plan.store_path(d))
            for v in part['versions']:
                for cond in [None] + part['conds']:
                    total += len(seeds)
                    if store is not None:
                        done += sum(store.has('%s/%s' % (v, part['weights']), label_of(cond), s) for s in seeds)
    return done, total


def show_tests(plan: Plan, name: str, say) -> float:
    t = plan.tests(name)
    sec = plan.raw.get('seconds_per_test') or {}
    done, total = _planned(plan, name)
    say('%s: %s (tests the runs of %s)' % (name, t.get('title', ''), t['of']))
    left = 0.0
    for part, d, ms, work in _jobs(plan, name):
        n = _evals(work)
        left += n * float(sec.get(d, 2.5)) / 60
        say('  %-12s %-12s %2d runs to test, %4d model evaluations left' % (part['name'], d, len(ms), n))
    say('%d of %d scores on disk; about %s of testing left for the runs that are finished' % (done, total, hm(left)))
    return left


def test(plan: Plan, name: str, a) -> int:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    lock = single_instance(plan.out_root / 'running.lock')
    if lock is None:
        print('another formal.py is working in %s; this one stops' % plan.out_root)
        return 2
    say = Tee(plan.out_root / 'log.txt')
    t = plan.tests(name)
    subset = SMOKE['eval.subset'] if plan.smoke else 0
    sec = {d: float(v) for d, v in (plan.raw.get('seconds_per_test') or {}).items()}
    measured = {}                                       # dataset -> [seconds, evaluations] of this process
    say('=' * 100)
    say('%s  formal.py test %s%s | %s | torch %s | %s' % (
        time.strftime('%Y-%m-%d %H:%M'), name, ' --smoke' if plan.smoke else '', device_name, torch.__version__,
        plan.out_root))
    show_tests(plan, name, say)
    jobs = _jobs(plan, name)

    def per_eval(d):
        if d in measured and measured[d][1]:
            return measured[d][0] / measured[d][1]
        return sec.get(d, 2.5) * (0.1 if plan.smoke else 1.0)

    def rest(j, i):
        """Seconds left after condition i of job j."""
        s = _evals(jobs[j][3][i + 1:]) * per_eval(jobs[j][1])
        return s + sum(_evals(w) * per_eval(d) for _, d, _, w in jobs[j + 1:])

    failed, bench, stores, scored, start = {}, None, {}, 0, time.time()
    keep_awake(True)
    try:
        for j, (part, d, ms, work) in enumerate(jobs):
            if not ms:
                say('')
                say('-- %s | %s: no finished run to test' % (part['name'], d))
                continue
            if not work:
                continue
            try:
                if bench is None or bench.dataset != d:
                    bench = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    bench = Bench.open(d, ms[0].cfg['data'], plan.root,
                                       find_data_root(ms[0].cfg, plan.root, a.data_root), device,
                                       chunk=ms[0].cfg['eval']['chunk'], subset=subset)
                other = ['%s (%s)' % (m.key, ', '.join(bench.fits(m))) for m in ms if bench.fits(m)]
                if other:
                    raise RuntimeError('these models use another split or preprocessing: %s' % '; '.join(sorted(set(other))))
                if d not in stores:
                    stores[d] = Store(plan.store_path(d), bench.header())
                store = stores[d]
                zone = None
                if d in (t.get('zones') or {}):
                    zone = load_zones(plan.root / t['zones'][d], bench)
                todo = list({id(m): m for _, missing in work for m in missing}.values())
                for m in todo:
                    bench.load(m)
                    store.describe(m.key, version=m.key.split('/')[0], weights=part['weights'],
                                   folder=str(m.ckpt.parent.parent))
                say('')
                say('== %s | %s | %s | %d test pixels | %d runs, %d conditions, %d model evaluations'
                    % (part['name'], d, ' + '.join(part['suites']), len(bench.test), len(todo), len(work),
                       _evals(work)))

                def on_clean(m, pred, store=store, zone=zone, bench=bench):
                    store.put_zones(m.key, m.seed, zone_scores(pred, bench.y, zone, bench.scene.n_classes))

                for i, (cond, missing) in enumerate(work):
                    # an earlier part of this stage may have scored the condition in the meantime (the clean scene)
                    missing = [m for m in missing if not store.has(m.key, label_of(cond), m.seed)]
                    if not missing:
                        continue
                    t0 = time.time()
                    res = bench.score(missing, cond, on_clean if zone is not None else None)
                    for m, s in zip(missing, res):
                        store.put(m.key, label_of(cond), m.seed, s)
                    store.save()
                    scored += len(missing)
                    got = measured.setdefault(d, [0.0, 0])
                    got[0] += time.time() - t0
                    got[1] += _evals([(cond, missing)])
                    by_key = {}
                    for m, s in zip(missing, res):
                        by_key.setdefault(m.key.split('/')[0], []).append(s['OA'])
                    left = rest(j, i)
                    say('   [%3d/%d] %-24s %s | the rest about %s (until about %s)' % (
                        i + 1, len(work), label_of(cond),
                        '  '.join('%s %.2f' % (k, np.mean(v)) for k, v in by_key.items()),
                        hm(left / 60), time.strftime('%m-%d %H:%M', time.localtime(time.time() + left))))
                # the clean scores must repeat the training records (same weights, same test pixels)
                bad = ['%s seed %d: %.3f here, %.3f in training' % (m.key, m.seed, got_oa, m.recorded)
                       for m in todo if m.recorded is not None and m.cfg['eval'].get('subset', 0) == subset
                       for got_oa in [store.get(m.key, CLEAN)[m.seed]['OA']] if abs(got_oa - m.recorded) > 0.02]
                store.data.setdefault('clean_check', {})[part['name']] = bad
                store.save()
                say('   clean OA ' + ('repeats the training records' if not bad else 'DIFFERS: ' + '; '.join(bad)))
                for m in todo:
                    m.model = None
            except KeyboardInterrupt:
                raise
            except Exception as e:           # noqa: BLE001 - a part that fails must not stop the other parts
                failed[(part['name'], d)] = '%s: %s' % (type(e).__name__, e)
                say('!! %s on %s failed (%s); going on with the next part' % (part['name'], d, failed[(part['name'], d)]))
                say(traceback.format_exc())
                bench = None
    except KeyboardInterrupt:
        say('')
        say('stopped by the user: the scores on disk are kept, start again to go on')
    finally:
        keep_awake(False)
    say('')
    say('scored %d runs under conditions in %s' % (scored, hm((time.time() - start) / 60)))
    text = tests_summary(plan, name)
    if failed:
        text += '\n' + '\n'.join('!! FAILED: %s on %s (%s)' % (p, d, why) for (p, d), why in failed.items())
    say('=' * 100)
    say(text)
    say('=' * 100)
    say('(%s)' % publish(plan.out_root / ('summary_%s.txt' % name), text, clipboard=not plan.smoke))
    say('(every record without the weights: %s)' % pack(plan))
    if lock:
        lock.close()
    return 1 if failed else 0


def _cell(value, width=8, fmt='%.2f') -> str:
    return '%*s' % (width, '-' if value is None else fmt % value)


def _complete(store, key, labels) -> list:
    """The seeds of a model that have a score under every one of the conditions."""
    return [] if store is None else mean_over_seeds(store, key, labels)[2]


def tests_summary(plan: Plan, name: str) -> str:
    t = plan.tests(name)
    st = plan.stage(t['of'])
    n_seeds = len(st.get('seeds', plan.seeds))
    stores = {d: Store.read(plan.store_path(d)) for d in st['datasets']}
    done, total = _planned(plan, name)
    lines = ['SUMMARY of %s (%s): %s' % (name, plan.exp, t.get('title', '')),
             'OA in %, mean over the seeds that have every condition of the table; a condition with several draws '
             'counts with the mean of its draws.']

    def mean(store, key, labels, seeds):
        return None if not seeds else mean_over_seeds(store, key, labels, seeds=seeds)[0]

    for part in plan.parts(name):
        w = part['weights']
        lines += ['', '#### part %s: weights of the %s epoch' % (part['name'], w)]
        corr = [c for c in part['conds'] if not isinstance(c, (Shift, Refill))]
        shifts = [c for c in part['conds'] if isinstance(c, Shift)]
        refills = [c for c in part['conds'] if isinstance(c, Refill)]
        for d in st['datasets']:
            store = stores[d]
            if store is None:
                lines += ['', '%s: nothing yet' % d]
                continue
            keys = [(v, '%s/%s' % (v, w)) for v in part['versions']]
            if corr:
                tests = list(dict.fromkeys(c.test for c in corr))
                labels = [c.label for c in corr]
                lines += ['', '%s, %d test pixels: the %d corrupted conditions' % (d, store.data['n_test'], len(corr)),
                          '%-13s %5s %8s' % ('version', 'runs', 'clean') + ''.join('%8s' % x for x in tests)
                          + '%15s %8s' % ('all (± std)', 'drop')]
                for v, key in keys:
                    seeds = _complete(store, key, [CLEAN] + labels)
                    clean = mean(store, key, [CLEAN], seeds)
                    allm, per_seed, _ = mean_over_seeds(store, key, labels, seeds=seeds) if seeds else (None, [], [])
                    lines.append('%-13s %2d/%-2d %s' % (v, len(seeds), n_seeds, _cell(clean)) + ''.join(
                        _cell(mean(store, key, [c.label for c in corr if c.test == x], seeds)) for x in tests)
                        + ('%15s %8s' % ('%.2f ± %.2f' % (allm, np.std(per_seed)), '%+.2f' % (allm - clean))
                           if seeds else '%15s %8s' % ('-', '-')))
                if w == 'best':
                    groups = list(dict.fromkeys((c.test, c.target) for c in corr))
                    lines += ['', '%s: mean OA per test and corrupted modality' % d,
                              '%-13s' % 'version' + ''.join('%14s' % ('%s %s' % g) for g in groups)]
                    for v, key in keys:
                        seeds = _complete(store, key, [CLEAN] + labels)
                        lines.append('%-13s' % v + ''.join(
                            _cell(mean(store, key, [c.label for c in corr if (c.test, c.target) == g], seeds), 14)
                            for g in groups))
            if shifts:
                labels = [c.label for c in shifts]
                lines += ['', '%s: LiDAR shifted against the HSI (mean of the four directions)' % d,
                          '%-13s %5s %8s' % ('version', 'runs', 'clean') + ''.join('%8s' % ('%g px' % c.level)
                                                                                   for c in shifts)
                          + '%14s' % ('drop at %g px' % shifts[0].level)]
                for v, key in keys:
                    seeds = _complete(store, key, [CLEAN] + labels)
                    clean, first = mean(store, key, [CLEAN], seeds), mean(store, key, labels[:1], seeds)
                    lines.append('%-13s %2d/%-2d %s' % (v, len(seeds), n_seeds, _cell(clean)) + ''.join(
                        _cell(mean(store, key, [lb], seeds)) for lb in labels)
                        + '%14s' % ('%+.2f' % (first - clean) if seeds else '-'))
            if refills:
                kinds = list(dict.fromkeys((c.test, c.target) for c in refills))
                lines += ['', '%s: lost values filled by the band means or by nearest-neighbour interpolation '
                              '(mean over the levels)' % d,
                          '%-13s %5s' % ('version', 'runs') + ''.join('%22s%22s' % ('%s %s, mean' % k, '%s %s, nearest' % k)
                                                                     for k in kinds)]
                for v, key in keys:
                    both = [c.base.label for c in refills] + [c.label for c in refills]
                    seeds = _complete(store, key, both)
                    lines.append('%-13s %2d/%-2d' % (v, len(seeds), n_seeds) + ''.join(
                        _cell(mean(store, key, [c.base.label for c in refills if (c.test, c.target) == k], seeds), 22)
                        + _cell(mean(store, key, [c.label for c in refills if (c.test, c.target) == k], seeds), 22)
                        for k in kinds))
        # zones: scored on the clean scene of every run that this part tested
        for d in st['datasets']:
            store = stores[d]
            if store is None or d not in (t.get('zones') or {}) or not corr:
                continue
            cols = ('all', 'shadow', 'outside', 'outside*', 'cost')
            lines += ['', '%s: inside and outside the cloud shadow, clean test pixels (outside* = the outside pixels '
                          'with the class mix of the shadow; cost = outside* - shadow)' % d,
                      '%-13s %5s' % ('version', 'runs') + ''.join('%16s' % c for c in cols)]
            for v in part['versions']:
                z = store.zones('%s/%s' % (v, w))
                lines.append('%-13s %2d/%-2d' % (v, len(z), n_seeds) + ''.join(
                    '%16s' % ('%.2f ± %.2f' % (np.mean([s[c] for s in z.values()]), np.std([s[c] for s in z.values()]))
                              if z else '-') for c in cols))
    lines += ['', '%d of %d scores on disk.' % (done, total)]
    for d, store in stores.items():
        for part_name, bad in ((store.data.get('clean_check') or {}) if store else {}).items():
            if bad:
                lines.append('!! %s, part %s: the clean OA differs from the training records: %s'
                             % (d, part_name, '; '.join(bad)))
    return '\n'.join(lines + checks(plan, name, stores))


def checks(plan: Plan, name: str, stores: dict) -> list:
    """The rules of a test stage (configs/formal.yaml, written down before its first run), applied to what is on
    disk. B and A are the two versions of `pair`, with the weights of the best epoch."""
    t = plan.tests(name)
    rules = t.get('rules') or {}
    if not rules:
        return []
    datasets = plan.stage(t['of'])['datasets']
    b, a_ = rules['pair']
    kb, ka = b + '/best', a_ + '/best'
    conds = [c for part in plan.parts(name) for c in part['conds']]
    lines = ['', 'checks (rules written down before the first run; B = %s, A = %s, weights of the best epoch)' % (b, a_)]

    def oa(d, key, labels):
        return None if stores[d] is None or not labels else mean_over_seeds(stores[d], key, labels)[0]

    def per_dataset(values):
        return ', '.join('%s %s' % (d, '-' if v is None else '%+.2f' % v) for d, v in values.items())

    def drop(d, key, label):
        """Clean OA minus the OA under a condition, over the seeds that have both."""
        seeds = _complete(stores[d], key, [CLEAN, label])
        if not seeds:
            return None
        return (mean_over_seeds(stores[d], key, [CLEAN], seeds=seeds)[0]
                - mean_over_seeds(stores[d], key, [label], seeds=seeds)[0])

    r = rules.get('shift')
    if r:
        label = Shift(float(r['level'])).label
        ex = {}
        for d in datasets:
            db, da = drop(d, kb, label), drop(d, ka, label)
            ex[d] = None if None in (db, da) else db - da
        vals = list(ex.values())
        if None in vals:
            verdict = 'NOT EVALUATED (scores missing)'
        elif all(v <= r['same'] for v in vals):
            verdict = 'HOLDS: B is not more sensitive to the shift than A'
        elif sum(v > r['worse'] for v in vals) >= 2:
            verdict = 'FAILS: B is more sensitive to the shift than A'
        else:
            verdict = 'NO CLAIM'
        lines.append('shift of %g px, drop of B minus drop of A: %s. Holds if at most +%g on every dataset, fails if '
                     'above +%g on two or more -> %s' % (r['level'], per_dataset(ex), r['same'], r['worse'], verdict))
    r = rules.get('interp')
    if r:
        for kind in dict.fromkeys((c.test, c.target) for c in conds if isinstance(c, Refill)):
            mine = [c for c in conds if isinstance(c, Refill) and (c.test, c.target) == kind]
            mean_fill, nearest = [c.base.label for c in mine], [c.label for c in mine]
            g, h = {}, {}
            for d in datasets:
                bm, an, bn = oa(d, kb, mean_fill), oa(d, ka, nearest), oa(d, kb, nearest)
                g[d] = None if None in (bm, an) else bm - an
                h[d] = None if None in (bn, an) else bn - an
            vals = list(g.values())
            if None in vals:
                verdict = 'NOT EVALUATED (scores missing)'
            elif sum(v >= r['better'] for v in vals) >= 2 and min(vals) >= -r['same']:
                verdict = 'HOLDS: B with the band means is better than A with interpolation'
            elif sum(v <= -r['better'] for v in vals) >= 2:
                verdict = 'FAILS: A with interpolation is better than B with the band means'
            else:
                verdict = 'NO CLAIM'
            lines.append('%s %s, B with the band means minus A with interpolation: %s. Holds if at least +%g on two '
                         'or more datasets and none below -%g, fails if -%g or lower on two or more -> %s'
                         % (kind[0], kind[1], per_dataset(g), r['better'], r['same'], r['better'], verdict))
            lines.append('%s %s, both with interpolation (the gate does not respond), B minus A: %s'
                         % (kind[0], kind[1], per_dataset(h)))
    r = rules.get('shadow')
    if r:
        for d in (t.get('zones') or {}):
            zb, za = (stores[d].zones(kb), stores[d].zones(ka)) if stores.get(d) else ({}, {})
            if len(zb) < 2 or len(za) < 2:
                lines.append('cloud shadow of %s: NOT EVALUATED (fewer than two runs on one side)' % d)
                continue
            vb, va = [s['shadow'] for s in zb.values()], [s['shadow'] for s in za.values()]
            gain = float(np.mean(vb) - np.mean(va))
            se = float(np.sqrt(np.var(vb, ddof=1) / len(vb) + np.var(va, ddof=1) / len(va)))
            clear = se > 0 and abs(gain) > 2 * se
            if gain >= r['better'] and clear:
                verdict = 'B IS CLEARLY BETTER IN THE SHADOW'
            elif gain <= -r['better'] and clear:
                verdict = 'B IS CLEARLY WORSE IN THE SHADOW'
            else:
                verdict = 'NO CLEAR DIFFERENCE (below %g points or within two standard errors)' % r['better']
            lines.append('cloud shadow of %s, OA on the shadow pixels, B minus A: %+.2f (standard error %.2f) -> %s'
                         % (d, gain, se, verdict))
    return lines


def timing(plan: Plan, a) -> int:
    """Training time per epoch and test time per patch of the versions under `timing` in configs/formal.yaml, on
    this machine, with nothing else of formal.py running. Training: two short trainings of different length; the
    difference of their times divided by the difference of their epochs leaves out the start-up and the scoring
    of the test pixels. Test: the test-time network on all test pixels, the fastest of three passes after a
    warm-up pass. The weights do not matter for the times, so nothing is loaded and nothing is saved but
    results/<exp>/timing.json."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    device_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'
    lock = single_instance(plan.out_root / 'running.lock')
    if lock is None:
        print('another formal.py is working in %s; this one stops' % plan.out_root)
        return 2
    say = Tee(plan.out_root / 'log.txt')
    t = plan.raw['timing']
    short, long_ = (1, 2) if plan.smoke else t['epochs']
    say('=' * 100)
    say('%s  formal.py timing%s | %s | torch %s' % (time.strftime('%Y-%m-%d %H:%M'), ' --smoke' if plan.smoke else '',
                                                    device_name, torch.__version__))

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    rows, scenes = [], {}
    keep_awake(True)
    try:
        for d in t['datasets']:
            for v in t['versions']:
                cfg = plan.config(d, v)
                st = Setting(cfg, d, plan.root, find_data_root(cfg, plan.root, a.data_root), device, scenes)
                took = []
                for epochs in (short, long_):                # the test pixels are scored once, after the last epoch
                    c = apply(cfg, [('training.epochs', epochs), ('eval.every', epochs), ('print_every', 10 ** 6)])
                    took.append(train_run(c, st.scene, st.sampler, st.split, st.pool, 0, device,
                                          log=lambda *_: None)['seconds'])
                per_epoch = (took[1] - took[0]) / (long_ - short)
                model = build_model(cfg, st.scene).strip_training_branch().to(device).eval()
                idx = torch.from_numpy(st.split['test']).to(device)
                if cfg['eval'].get('subset'):
                    idx = idx[:cfg['eval']['subset']]
                passes = []
                for i in range(4):
                    sync()
                    t0 = time.time()
                    predict(model, st.sampler, idx, cfg['eval']['chunk'])
                    sync()
                    passes.append(time.time() - t0)
                best = min(passes[1:])
                rows.append(dict(dataset=d, version=v, train_s_per_epoch=per_epoch, test_ms_per_patch=1000 * best / len(idx),
                                 test_s_all=best, n_test=int(len(idx)), train_seconds=took, epochs=[short, long_],
                                 steps_per_epoch=-(-len(st.split['train']) // cfg['training']['batch_size'])))
                say('   %-12s %-6s training %7.3f s per epoch | test %7.4f ms per patch (%.2f s for %d patches)'
                    % (d, v, per_epoch, rows[-1]['test_ms_per_patch'], best, len(idx)))
                del model
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    finally:
        keep_awake(False)
    record = dict(device=device_name, torch=torch.__version__, created=time.strftime('%Y-%m-%d %H:%M'), rows=rows)
    path = plan.out_root / 'timing.json'
    path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    lines = ['SUMMARY of timing (%s) on %s: seconds per training epoch, milliseconds per test patch' % (plan.exp, device_name),
             '%-13s %-8s %18s %18s %14s' % ('dataset', 'version', 'train s/epoch', 'test ms/patch', 'test s (all)')]
    lines += ['%-13s %-8s %18.3f %18.4f %14.2f' % (r['dataset'], r['version'], r['train_s_per_epoch'],
                                                   r['test_ms_per_patch'], r['test_s_all']) for r in rows]
    text = '\n'.join(lines)
    say('=' * 100)
    say(text)
    say('=' * 100)
    say('(%s)' % publish(plan.out_root / 'summary_timing.txt', text, clipboard=not plan.smoke))
    say('(every record without the weights: %s)' % pack(plan))
    if lock:
        lock.close()
    return 0


def maps(plan: Plan, a) -> int:
    """The predicted class of every pixel of every scene (uint8, 255 = not predicted), for the figures of the
    paper: results/<exp>/maps/<dataset>/<version>_seed<k>_<condition>.npz. The versions and conditions are listed
    under `maps` in configs/formal.yaml: the clean scene for every finished seed, the conditions (their first
    draw) for the first seed. Maps on disk are not made again. --smoke predicts the test pixels only."""
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    lock = single_instance(plan.out_root / 'running.lock')
    if lock is None:
        print('another formal.py is working in %s; this one stops' % plan.out_root)
        return 2
    say = Tee(plan.out_root / 'log.txt')
    t = plan.raw['maps']
    st = plan.stage(t['of'])
    known = {c.label: c for c in suite('corruptions')}
    conds = [known[lb] for lb in t['conditions']]
    part = dict(versions=list(t['versions']), weights='best')
    say('=' * 100)
    say('%s  formal.py maps%s | %s' % (time.strftime('%Y-%m-%d %H:%M'), ' --smoke' if plan.smoke else '', plan.out_root))
    made, failed = 0, {}
    keep_awake(True)
    try:
        for d in st['datasets']:
            try:
                ms = plan.members(t['of'], part, d)
                if not ms:
                    say('-- %s: no finished run' % d)
                    continue
                out = plan.out_root / 'maps' / d
                first = min(m.seed for m in ms)
                jobs = [(m, cond) for m in ms for cond in [None] + (conds if m.seed == first else [])]
                jobs = [(m, c) for m, c in jobs if not _map_path(out, m, c).exists()]
                if not jobs:
                    continue
                bench = Bench.open(d, ms[0].cfg['data'], plan.root, find_data_root(ms[0].cfg, plan.root, a.data_root),
                                   device, chunk=ms[0].cfg['eval']['chunk'],
                                   subset=SMOKE['eval.subset'] if plan.smoke else 0)
                H, W = bench.scene.shape
                idx = bench.test_idx if plan.smoke else torch.arange(H * W, device=device)
                out.mkdir(parents=True, exist_ok=True)
                say('== %s | %d maps of %d x %d pixels' % (d, len(jobs), H, W))
                for cond in [None] + conds:                  # one test scene, every model that needs it
                    mine = [m for m, c in jobs if c is cond]
                    if not mine:
                        continue
                    view = bench.scene if cond is None else apply_condition(bench.scene, cond, bench.stats, bench.seed,
                                                                            bench.block, 0)
                    samplers = {}
                    for m in mine:
                        if m.model is None:
                            bench.load(m)
                        key = (m.cfg['data']['patch'], m.cfg['data']['padding'])
                        if key not in samplers:
                            samplers[key] = PatchSampler(view, key[0], device, key[1])
                        pred = predict(m.model, samplers[key], idx, bench.chunk).cpu().numpy()
                        full = np.full(H * W, 255, np.uint8)
                        full[idx.cpu().numpy()] = pred
                        oa = 100 * float((full[bench.test] == bench.y).mean())
                        path = _map_path(out, m, cond)
                        tmp = path.with_name(path.name + '.tmp.npz')
                        np.savez_compressed(tmp, pred=full.reshape(H, W), version=m.key.split('/')[0], seed=m.seed,
                                            condition=label_of(cond), test_oa=oa)
                        os.replace(tmp, path)
                        made += 1
                        say('   %-34s test OA %.2f' % (path.name, oa))
                    del samplers, view
                for m in ms:
                    m.model = None
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except KeyboardInterrupt:
                raise
            except Exception as e:           # noqa: BLE001 - one dataset that fails must not stop the others
                failed[d] = '%s: %s' % (type(e).__name__, e)
                say('!! the maps of %s failed (%s)' % (d, failed[d]))
                say(traceback.format_exc())
    except KeyboardInterrupt:
        say('stopped by the user: the maps on disk are kept')
    finally:
        keep_awake(False)
    say('made %d maps in %s' % (made, plan.out_root / 'maps'))
    say('(every record without the weights: %s)' % pack(plan))
    if lock:
        lock.close()
    return 1 if failed else 0


def _map_path(out: Path, m, cond) -> Path:
    return out / ('%s_seed%d_%s.npz' % (m.key.split('/')[0], m.seed, label_of(cond).replace(' ', '_')))


def pack(plan: Plan) -> Path:
    """results/SymbioNet_<exp>_results.zip: every record, summary and log of the experiment, without the weights
    (the file to send back; the tables and figures of the paper are made from it)."""
    out = plan.root / 'results' / ('SymbioNet_%s_results.zip' % plan.exp)
    tmp = out.with_name(out.name + '.tmp')
    with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(plan.out_root.rglob('*')):
            if f.is_file() and f.suffix not in ('.pt', '.tmp', '.lock'):
                z.write(f, f.relative_to(plan.out_root.parent).as_posix())
    os.replace(tmp, out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('plan', 'run', 'test', 'summary', 'timing', 'maps', 'pack'))
    p.add_argument('stage', nargs='?', help='a training or test stage of configs/formal.yaml (plan and summary: all '
                                            'of them if left out)')
    p.add_argument('--data-root', help='folder with Trento/, Houston13/, MUUFL/ (default: data.root)')
    p.add_argument('--device', help='cuda, cpu, ... (default: cuda if available)')
    p.add_argument('--limit', type=int, default=0, help='run: train at most this many runs, then stop')
    p.add_argument('--seeds', help='train only these seeds of the plan, e.g. 0,1 (to share a stage between machines)')
    p.add_argument('--smoke', action='store_true',
                   help='2 epochs on part of the test set, saved under results/formal_smoke: checks the setup')
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:                    # noqa: BLE001
        pass
    plan = Plan(smoke=a.smoke)
    trainings, testings = list(plan.raw['stages']), list(plan.raw.get('tests') or {})
    if a.command == 'pack':
        print(pack(plan))
        return 0
    if a.command == 'timing':
        return timing(plan, a)
    if a.command == 'maps':
        return maps(plan, a)
    if a.command in ('run', 'test'):
        mine, other = (trainings, testings) if a.command == 'run' else (testings, trainings)
        if a.stage in other:
            raise SystemExit('%s is a %s stage: python formal.py %s %s'
                             % (a.stage, 'test' if a.command == 'run' else 'training',
                                'test' if a.command == 'run' else 'run', a.stage))
        if a.stage not in mine:
            raise SystemExit('%s needs one of: %s' % (a.command, ', '.join(mine) or 'nothing is planned'))
        return run(plan, a.stage, a) if a.command == 'run' else test(plan, a.stage, a)
    for s in ([a.stage] if a.stage else trainings + testings):
        if a.command == 'plan':
            show_tests(plan, s, print) if plan.is_test(s) else show_plan(plan, s, print)
        else:
            print(tests_summary(plan, s) if plan.is_test(s) else summary(plan, s))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
