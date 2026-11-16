#!/usr/bin/env python3
"""Published networks under the test conditions of this project (see symbionet/external.py).

    python compare.py check      the four sources are found under external/, each network is built and run once on
                                 random patches (nothing is trained, no dataset is read, the CPU is enough)
    python compare.py all        wait until nothing else is working in results/formal, then: train what is missing,
                                 test it under the 46 conditions, print the summary, pack the results
    python compare.py run | test | summary        one step of `all`
    options: --methods dahgm,cnnh   --datasets muufl   --seeds 0,1   --limit N   --external FOLDER   --data-root
             FOLDER   --device cpu   --no-wait   --smoke (2 epochs, part of the test pixels, results/formal_smoke)

What is run: the methods of symbionet/external.py (dahgm, cnnh, endnet, mft), on the three datasets, with the seeds
of the plan (configs/formal.yaml). Every run uses the settings of the version `base` of the plan, i.e. the protocol
of the base network: the split, the patch size, the normalisation, the optimizer and its schedule, the epochs, the
batch size, the augmentation, label smoothing and class weights. The network, its own preprocessing, its loss terms
and its prediction rule are those of the published code.

Where it goes: the runs of a method are kept like those of a version of the plan, under the name ext_<method>
(results/formal/<dataset>/ext_<method>/: config.yaml, run<k>.json, ckpt/, summary.json), and their test scores in
the score file of the dataset (results/formal/tests/<dataset>.json) under the key ext_<method>/best. So the tools
that read the results of the plan read these too, and `formal.py pack` packs them.

This file changes no file of the project and uses the lock of formal.py, so it never works at the same time as a
stage of the plan. `all`, `run` and `test` wait for their turn: while a package or a formal.py is working in
results/formal, and after that until the log of the plan has been silent for 25 minutes, so that a package that is
waiting there takes over first. Its own log is results/formal/log_compared.txt. Finished work is never repeated:
it can be stopped and started again at any time.
"""
import argparse
import os
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import formal  # noqa: E402
from formal import Plan, Tee, code_id, hm, keep_awake, pack, publish, same_settings, single_instance, thin  # noqa: E402
from symbionet import external  # noqa: E402
from symbionet.config import load_yaml  # noqa: E402
from symbionet.conditions import PAPER_TESTS, n_draws, suite  # noqa: E402
from symbionet.runs import (Setting, _atomic_json, find_data_root, mean_std, read_run, write_config,  # noqa: E402
                            write_summary)
from symbionet.testing import CLEAN, Bench, Member, Store, label_of, mean_over_seeds  # noqa: E402

PROTOCOL = 'base'                                        # the version of the plan whose settings every run uses
REFERENCES = (('base', '(a) Base'), ('full', 'SymbioNet'))   # versions of the plan shown next to the methods
POLL, REPORT = 30, 600                                   # waiting for the queue: look / report every ... seconds
QUIET = 1500     # s; the log of the plan must be silent this long before this file takes its turn. A package that
#                  waits for the one before it takes over within 930 s of that one's last line (go_formal.py).
LOG = 'log_compared.txt'                                 # not log.txt: a waiting package watches that file


def version(method: str) -> str:
    return 'ext_' + method


def locked(path: Path) -> bool:
    """True while another process holds a lock file of the project (as in the launchers of the packages)."""
    if not path.exists():
        return False
    try:
        f = open(path, 'a+')
    except OSError:
        return True
    try:
        if os.name == 'nt':
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        f.close()


def take_lock(plan: Plan, wait: bool, quiet: float = QUIET, poll: float = POLL):
    """The lock of formal.py (running.lock) for this process.
    wait: waits while a package (it holds launcher.lock) or a formal.py (running.lock) is working there, and after
    that until nothing has worked there for `quiet` seconds (the locks free, the log of the plan silent), so that a
    package that is waiting for its turn goes first. Try-out runs (--smoke) have their own folder and no queue.
    not wait: the lock if nothing is working there now, else None."""
    log = plan.out_root / 'log.txt'
    if plan.smoke:
        quiet = 0
    seen_busy, last, state = 0.0, 0.0, None
    while True:
        if locked(plan.out_root / 'launcher.lock') or locked(plan.out_root / 'running.lock'):
            seen_busy, now = time.time(), 'busy'
        else:
            newest = max(seen_busy, log.stat().st_mtime if log.exists() else 0.0)
            rest = quiet - (time.time() - newest) if wait else 0
            now = 'quiet' if rest > 0 else 'free'
        if now == 'free':
            lock = single_instance(plan.out_root / 'running.lock')
            if lock is not None:                         # None: taken by someone else in this very moment
                return lock
            seen_busy, now = time.time(), 'busy'
        if not wait:
            return None
        if now != state or time.time() - last >= REPORT:
            print('   %s  %s' % (time.strftime('%m-%d %H:%M'), (
                'something else is working in %s: waiting for it to finish (leave this window open)' % plan.out_root
                if now == 'busy' else
                'nothing is working there now; starting in about %s unless a waiting package takes over first'
                % hm(max(rest, 60) / 60))), flush=True)
            state, last = now, time.time()
        time.sleep(poll)


class Job:
    """What the commands share: the plan, the methods, datasets and seeds asked for, the device, the folders."""

    def __init__(self, a):
        self.a = a
        self.plan = Plan(smoke=a.smoke)
        known = list(external.METHODS)
        self.methods = [m.strip() for m in a.methods.split(',')] if a.methods else known
        bad = [m for m in self.methods if m not in known]
        if bad:
            raise SystemExit('unknown method(s) %s (choose from %s)' % (', '.join(bad), ', '.join(known)))
        every = list(self.plan.raw['datasets'])
        self.datasets = [d.strip() for d in a.datasets.split(',')] if a.datasets else every
        bad = [d for d in self.datasets if d not in every]
        if bad:
            raise SystemExit('unknown dataset(s) %s (choose from %s)' % (', '.join(bad), ', '.join(every)))
        self.seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else list(self.plan.seeds)
        self.ext = Path(a.external).resolve() if a.external else self.plan.root / 'external'
        self.device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else 'cpu'

    def cfg(self, dataset: str) -> dict:
        return self.plan.config(dataset, PROTOCOL)

    def folder(self, dataset: str, method: str) -> Path:
        return self.plan.folder(dataset, version(method))

    def done(self, dataset: str, method: str, seed: int):
        res = read_run(self.folder(dataset, method), seed)
        return res if res is not None and res.get('seed') == seed else None

    def runs(self) -> list:
        """[(dataset, method, seed)] in the order of training: method by method, the datasets inside."""
        return [(d, m, s) for m in self.methods for d in self.datasets for s in self.seeds]


# --------------------------------------------------------------------------------------------------------- check
def check(job: Job) -> int:
    print('sources under %s' % job.ext)
    rng, failed = np.random.default_rng(0), 0
    shapes = ((144, 15, 11), (64, 11, 9), (63, 6, 9))   # bands, classes, patch side of the three datasets
    for m in job.methods:
        src = external.SOURCES[m]
        try:
            path, warning = external.locate(m, job.ext)
            print('\n%-7s %s\n        %s' % (m, src['name'], path))
            if warning:
                print('        !! ' + warning)
            for bands, classes, patch in shapes:
                scene = types.SimpleNamespace(hsi=rng.random((bands, 24, 24), dtype=np.float32),
                                              lidar=rng.random((1, 24, 24), dtype=np.float32),
                                              valid=np.ones((24, 24), bool), n_classes=classes)
                model = external.build(m, scene, patch, job.ext).to(job.device)
                h = torch.rand(6, bands, patch, patch, device=job.device)
                l = torch.rand(6, 1, patch, patch, device=job.device)
                y = torch.randint(0, classes, (6,), device=job.device)
                model.train()
                model.training_loss(h, l, y, torch.nn.CrossEntropyLoss()).backward()
                model.before_eval(lambda: [(h, l, y)])
                model.eval()
                with torch.no_grad():
                    out = model(h, l)
                if tuple(out.shape) != (6, classes) or not bool(torch.isfinite(out).all()):
                    raise RuntimeError('wrong output for %d bands, %d classes: %s' % (bands, classes, tuple(out.shape)))
            n = model.count_parameters()
            print('        built and run for the three datasets; %d parameters%s%s'
                  % (n['test_time'], ' (+ %d used in training only)' % n['training_only'] if n['training_only'] else '',
                     '; Mamba: %s' % model.info['mamba'] if 'mamba' in model.info else ''))
        except Exception as e:           # noqa: BLE001 - report every method
            failed += 1
            print('\n%-7s %s\n        !! %s: %s' % (m, src['name'], type(e).__name__, e))
            if not isinstance(e, FileNotFoundError):
                print(traceback.format_exc())
    print('\nSUMMARY of the check: %d of %d methods are ready%s' % (
        len(job.methods) - failed, len(job.methods),
        '' if not failed else '; download what is missing (addresses above) into %s and check again' % job.ext))
    return 1 if failed else 0


# ----------------------------------------------------------------------------------------------------------- run
def run(job: Job, say) -> int:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    plan, a = job.plan, job.a
    extra = dict(code=code_id(plan.root), device=job.device_name, torch=torch.__version__, protocol=PROTOCOL)
    say('=' * 100)
    say('%s  compare.py run%s | %s | torch %s | %s' % (time.strftime('%Y-%m-%d %H:%M'), ' --smoke' if plan.smoke else '',
                                                      job.device_name, torch.__version__, plan.out_root))
    say('published networks with the protocol of %r: %s' % (PROTOCOL, ', '.join(job.methods)))
    pending = [r for r in job.runs() if job.done(*r) is None]
    say('%d of %d runs are on disk' % (len(job.runs()) - len(pending), len(job.runs())))
    if a.limit:
        pending = pending[:a.limit]
    minutes = {}                                         # (dataset, method) -> minutes of the runs finished here
    for d, m, s in job.runs():
        res = job.done(d, m, s)
        if res and res.get('device') == job.device_name:
            minutes.setdefault((d, m), []).append(res['seconds'] / 60)

    def guess(d, m):
        if minutes.get((d, m)):
            return float(np.mean(minutes[(d, m)]))
        known = [v for k, vs in minutes.items() if k[1] == m for v in vs] or [v for vs in minutes.values() for v in vs]
        return float(np.mean(known)) if known else (1.0 if plan.smoke else 8.0)

    scenes, failed, trained, start = {}, {}, 0, time.time()
    st = group = None
    keep_awake(True)
    try:
        for i, (d, m, s) in enumerate(pending):
            if (d, m) in failed:
                continue
            try:
                cfg, out = job.cfg(d), job.folder(d, m)
                if group != (d, m):
                    st = group = None
                    if job.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    saved = dict(cfg, compared=dict(method=m, protocol=PROTOCOL,
                                                    repository=external.SOURCES[m]['repo']))
                    if (out / 'config.yaml').exists() and any(job.done(d, m, x) for x in plan.seeds):
                        diff = same_settings(load_yaml(out / 'config.yaml'), saved)
                        if diff:
                            raise RuntimeError('the runs on disk were trained with other settings (%s); move %s '
                                               'away to train this method again' % (', '.join(diff), out))
                    st = Setting(cfg, d, plan.root, find_data_root(cfg, plan.root, a.data_root), job.device, scenes)
                    group = (d, m)
                    write_config(out, saved)
                    say('')
                    say('== %s | %s (%s) | %s' % (d, version(m), external.SOURCES[m]['name'], st.scene.summary()))
                    say('   ' + st.describe())
                left = sum(guess(dd, mm) for dd, mm, _ in pending[i:] if (dd, mm) not in failed)
                say('[%3d/%d] %s %s seed %d | %s | this run about %s, the rest about %s (until about %s)'
                    % (i + 1, len(pending), d, version(m), s, time.strftime('%m-%d %H:%M'), hm(guess(d, m)), hm(left),
                       time.strftime('%m-%d %H:%M', time.localtime(time.time() + 60 * left))))
                res = external.train(cfg, m, st.scene, st.sampler, st.split, s, job.device, job.ext, log=say)
                if res['compared'].get('warning'):
                    say('   !! ' + res['compared']['warning'])
                res['n_train'] = int(len(st.split['train']))
                torch.save(res.pop('state'), out / 'ckpt' / ('run%d_best.pt' % s))
                torch.save(res.pop('last_state'), out / 'ckpt' / ('run%d_last.pt' % s))
                res.update(dataset=d, setting=version(m), run=s, split=st.split['path'], classes=st.scene.classes)
                res.update(extra, finished=time.strftime('%Y-%m-%d %H:%M'))
                _atomic_json(out / ('run%d.json' % s), res)
                say('   run %d (seed %d): best OA %.2f at epoch %d, last OA %.2f, AA %.2f, %.1f min'
                    % (s, s, res['best']['OA'], res['best_epoch'], res['last']['OA'], res['best']['AA'],
                       res['seconds'] / 60))
                trained += 1
                minutes.setdefault((d, m), []).append(res['seconds'] / 60)
                write_summary(out, [r for r in (job.done(d, m, x) for x in plan.seeds) if r], st.scene.classes)
            except KeyboardInterrupt:
                raise
            except Exception as e:           # noqa: BLE001 - a method that fails must not stop the others
                failed[(d, m)] = '%s: %s' % (type(e).__name__, e)
                say('!! %s %s failed (%s); going on with the next one' % (d, version(m), failed[(d, m)]))
                say(traceback.format_exc())
                st = group = None
    except KeyboardInterrupt:
        say('')
        say('stopped by the user: the finished runs are on disk, the interrupted run starts over next time')
        raise
    finally:
        keep_awake(False)
    say('')
    say('trained %d runs in %s' % (trained, hm((time.time() - start) / 60)))
    for (d, m), why in failed.items():
        say('!! FAILED: %s %s (%s)' % (d, version(m), why))
    return 1 if failed else 0


# ---------------------------------------------------------------------------------------------------------- test
def conditions(plan: Plan) -> list:
    conds = suite('corruptions')
    return [None] + (thin(conds) if plan.smoke else conds)


def test(job: Job, say) -> int:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan, a = job.plan, job.a
    subset = formal.SMOKE['eval.subset'] if plan.smoke else 0
    conds = conditions(plan)
    say('=' * 100)
    say('%s  compare.py test%s | %s | torch %s | %s' % (time.strftime('%Y-%m-%d %H:%M'), ' --smoke' if plan.smoke else '',
                                                       job.device_name, torch.__version__, plan.out_root))
    failed, scored, start = {}, 0, time.time()
    keep_awake(True)
    try:
        for d in job.datasets:
            try:
                cfg = job.cfg(d)
                members = [Member('%s/best' % version(m), s, load_yaml(job.folder(d, m) / 'config.yaml'),
                                  job.folder(d, m) / 'ckpt' / ('run%d_best.pt' % s), r['best']['OA'])
                           for m in job.methods for s in job.seeds for r in [job.done(d, m, s)] if r]
                store = Store.read(plan.store_path(d))
                work = [(c, [m for m in members if store is None or not store.has(m.key, label_of(c), m.seed)])
                        for c in conds]
                work = [(c, ms) for c, ms in work if ms]
                if not members:
                    say('')
                    say('-- %s: no finished run to test' % d)
                    continue
                if not work:
                    continue
                if job.device.type == 'cuda':
                    torch.cuda.empty_cache()
                bench = Bench.open(d, cfg['data'], plan.root, find_data_root(cfg, plan.root, a.data_root), job.device,
                                   chunk=cfg['eval']['chunk'], subset=subset)
                store = Store(plan.store_path(d), bench.header())
                todo = list({id(m): m for _, ms in work for m in ms}.values())
                other = ['%s (%s)' % (m.key, ', '.join(bench.fits(m))) for m in todo if bench.fits(m)]
                if other:
                    raise RuntimeError('these runs use another split or preprocessing: %s' % '; '.join(sorted(set(other))))
                for m in todo:
                    method = m.key.split('/')[0][len('ext_'):]
                    model = external.build(method, bench.scene, m.cfg['data']['patch'], job.ext)
                    model.load_state_dict(torch.load(m.ckpt, map_location='cpu', weights_only=True))
                    m.model = model.to(job.device).eval()
                    store.describe(m.key, version=m.key.split('/')[0], weights='best', folder=str(m.ckpt.parent.parent),
                                   method=external.SOURCES[method]['name'], protocol=PROTOCOL)
                evals = sum((1 if c is None else n_draws(c)) * len(ms) for c, ms in work)
                say('')
                say('== %s | %d test pixels | %d runs, %d conditions, %d model evaluations'
                    % (d, len(bench.test), len(todo), len(work), evals))
                t0, seen = time.time(), 0
                for i, (cond, ms) in enumerate(work):
                    res = bench.score(ms, cond)
                    for m, s in zip(ms, res):
                        store.put(m.key, label_of(cond), m.seed, s)
                    store.save()
                    scored += len(ms)
                    seen += (1 if cond is None else n_draws(cond)) * len(ms)
                    left = (time.time() - t0) / seen * (evals - seen)
                    by = {}
                    for m, s in zip(ms, res):
                        by.setdefault(m.key.split('/')[0], []).append(s['OA'])
                    say('   [%3d/%d] %-24s %s | the rest of %s about %s' % (
                        i + 1, len(work), label_of(cond), '  '.join('%s %.2f' % (k, np.mean(v)) for k, v in by.items()),
                        d, hm(left / 60)))
                bad = ['%s seed %d: %.3f here, %.3f in training' % (m.key, m.seed, got, m.recorded)
                       for m in todo for got in [store.get(m.key, CLEAN)[m.seed]['OA']] if abs(got - m.recorded) > 0.02]
                store.data.setdefault('clean_check', {})['compared'] = bad
                store.save()
                say('   clean OA ' + ('repeats the training records' if not bad else 'DIFFERS: ' + '; '.join(bad)))
                del bench, todo, members
            except KeyboardInterrupt:
                raise
            except Exception as e:           # noqa: BLE001 - a dataset that fails must not stop the others
                failed[d] = '%s: %s' % (type(e).__name__, e)
                say('!! the test on %s failed (%s); going on with the next dataset' % (d, failed[d]))
                say(traceback.format_exc())
    except KeyboardInterrupt:
        say('')
        say('stopped by the user: the scores on disk are kept, start again to go on')
        raise
    finally:
        keep_awake(False)
    say('')
    say('scored %d runs under conditions in %s' % (scored, hm((time.time() - start) / 60)))
    for d, why in failed.items():
        say('!! FAILED: the test on %s (%s)' % (d, why))
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------------- summary
def summary(job: Job) -> str:
    plan = job.plan
    conds = conditions(plan)[1:]
    groups = [(t, [label_of(c) for c in conds if c.test == t]) for t in PAPER_TESTS]
    every = [label_of(c) for c in conds]
    lines = ['SUMMARY of the compared methods (%s)%s' % (plan.exp, ', TRY-OUT RUNS: the numbers mean nothing' if plan.smoke else ''),
             'published networks trained here with the protocol of the base network; OA in %, mean over the seeds;',
             'clean = the epoch with the best test OA; the other columns = mean OA over the conditions of a test, and',
             'over all %d corrupted conditions (seeds that have every condition)' % len(every)]
    head = '%-13s %-22s %5s  %-13s %7s | %7s %7s %7s %7s %7s | %7s' % (
        'dataset', 'model', 'runs', 'clean', 'last', *[t for t, _ in groups], 'all')
    total = finished = tested = 0

    def cell(store, key, labels):
        value = mean_over_seeds(store, key, labels)[0] if store is not None and labels else None
        return '%7s' % ('-' if value is None else '%.2f' % value)

    for d in job.datasets:
        store = Store.read(plan.store_path(d))
        lines += ['', head, '-' * len(head)]
        for m in job.methods:
            runs = [r for r in (job.done(d, m, s) for s in job.seeds) if r]
            total += len(job.seeds)
            finished += len(runs)
            key = '%s/best' % version(m)
            name = external.SOURCES[m]['name']
            if not runs:
                lines.append('%-13s %-22s %2d/%-2d  (nothing yet)' % (d, name, 0, len(job.seeds)))
                continue
            tested += len(mean_over_seeds(store, key, every)[2]) if store is not None else 0
            oa, last = mean_std([r['best']['OA'] for r in runs]), mean_std([r['last']['OA'] for r in runs])
            lines.append('%-13s %-22s %2d/%-2d  %6.2f ± %-4.2f %7.2f | %s | %s' % (
                d, name, len(runs), len(job.seeds), oa[0], oa[1], last[0],
                ' '.join(cell(store, key, labels) for _, labels in groups), cell(store, key, every)))
        for v, name in REFERENCES:                        # the same columns for versions of the plan, if tested
            runs = [r for r in (plan.done(d, v, s) for s in plan.seeds) if r]
            if runs and store is not None:
                oa, last = mean_std([r['best']['OA'] for r in runs]), mean_std([r['last']['OA'] for r in runs])
                lines.append('%-13s %-22s %2d/%-2d  %6.2f ± %-4.2f %7.2f | %s | %s' % (
                    d, name, len(runs), len(plan.seeds), oa[0], oa[1], last[0],
                    ' '.join(cell(store, v + '/best', labels) for _, labels in groups), cell(store, v + '/best', every)))
    lines += ['', '%d of %d runs trained; %d of them tested under all %d conditions.' % (finished, total, tested, len(every))]
    warnings = sorted({r['compared']['warning'] for d in job.datasets for m in job.methods for s in job.seeds
                       for r in [job.done(d, m, s)] if r and r.get('compared', {}).get('warning')})
    lines += ['!! ' + w for w in warnings]
    mamba = sorted({r['compared']['mamba'] for d in job.datasets for s in job.seeds for r in [job.done(d, 'dahgm', s)]
                    if r and r.get('compared', {}).get('mamba')}) if 'dahgm' in job.methods else []
    if mamba:
        lines.append('Mamba blocks of DAHGM computed by: %s' % '; '.join(mamba))
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('check', 'all', 'run', 'test', 'summary'))
    p.add_argument('--methods', help='e.g. dahgm,cnnh (default: all four)')
    p.add_argument('--datasets', help='e.g. muufl,trento (default: the datasets of the plan)')
    p.add_argument('--seeds', help='e.g. 0,1 (default: the seeds of the plan)')
    p.add_argument('--limit', type=int, default=0, help='run: train at most this many runs, then go on')
    p.add_argument('--external', help='folder with the downloaded repositories (default: external/ in the project)')
    p.add_argument('--data-root', help='folder with Trento/, Houston13/, MUUFL/ (default: data.root)')
    p.add_argument('--device', help='cuda, cpu, ... (default: cuda if available)')
    p.add_argument('--no-wait', action='store_true', help='stop instead of waiting when something else is working')
    p.add_argument('--smoke', action='store_true',
                   help='2 epochs on part of the test set, saved under results/formal_smoke: checks the setup')
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:                    # noqa: BLE001
        pass
    job = Job(a)
    if a.command == 'check':
        return check(job)
    if a.command == 'summary':
        print(summary(job))
        return 0
    for m in job.methods:                                # fail before any waiting if a source is missing
        try:
            external.locate(m, job.ext)
        except FileNotFoundError as e:
            raise SystemExit('%s\n(python compare.py check shows what is there)' % e)
    if a.command == 'all' and check(job):                # before any waiting: every network can be built here
        print('nothing was started: the check above has to pass first')
        return 1
    try:
        lock = take_lock(job.plan, not a.no_wait)
    except KeyboardInterrupt:
        print('stopped while waiting; nothing was started')
        return 130
    if lock is None:
        print('something else is working in %s; this one stops (--no-wait)' % job.plan.out_root)
        return 2
    say = Tee(job.plan.out_root / LOG)
    rc = 0
    try:
        if a.command in ('all', 'run'):
            rc |= run(job, say)
        if a.command in ('all', 'test'):
            rc |= test(job, say)
    except KeyboardInterrupt:
        rc = 130
    text = summary(job)
    say('=' * 100)
    say(text)
    say('=' * 100)
    say('(%s)' % publish(job.plan.out_root / 'summary_compared.txt', text, clipboard=not job.plan.smoke))
    say('(every record without the weights: %s)' % pack(job.plan))
    if lock:
        lock.close()
    return rc


if __name__ == '__main__':
    sys.exit(main())
