"""Training runs on disk. One setting of one dataset lives in results/<exp>/<dataset>/<setting>/:

    config.yaml           the resolved settings
    run<k>.json           scores at the best and the last epoch, the curves; written last, so a run<k>.json on disk
                          means that run k is finished
    ckpt/run<k>_best.pt   weights at the epoch with the best test OA
    ckpt/run<k>_last.pt   weights after the last epoch
    summary.json          mean and standard deviation over the runs

train.py saves records using this module; evaluate.py reads them.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import PatchSampler, class_counts, get_split, load_scene, pool_indices, subset_train
from .engine import train_run
from .metrics import SCORES

try:
    import yaml
except ImportError:                     # pragma: no cover
    raise SystemExit('PyYAML is needed for the configuration files: pip install pyyaml')


def find_data_root(cfg: dict, root: Path, override=None) -> Path:
    """The dataset folder: `override`, or data.root (relative paths: under the working directory, then under root)."""
    data_root = Path(override or cfg['data']['root'])
    if not data_root.is_absolute():
        data_root = (Path.cwd() / data_root) if (Path.cwd() / data_root).exists() else Path(root) / data_root
    if not data_root.exists():
        raise SystemExit('dataset folder not found: %s (use --data-root)' % data_root)
    return data_root


class Setting:
    """What the runs of one setting share: the scene, the split, the unlabeled pool and the patch sampler."""

    def __init__(self, cfg: dict, dataset: str, root: Path, data_root: Path, device, scenes: dict = None):
        d = cfg['data']
        scenes = {} if scenes is None else scenes                 # scenes already loaded for this dataset
        key = (dataset, d['hsi_norm'], d['lidar_norm'], d['pca'], d['pca_scale'])
        if key not in scenes:
            scenes[key] = load_scene(dataset, data_root, d['hsi_norm'], d['lidar_norm'], d['pca'], d['pca_scale'],
                                     cfg['seed'])
        self.dataset, self.scene = dataset, scenes[key]
        self.split = get_split(self.scene, Path(root) / d['split_dir'], d['split'], d['split_ratio'], d['split_seed'],
                               d['split_file'])
        self.pool = pool_indices(self.scene, self.split, d.get('pool_test', False), d.get('pool_size', 0),
                                 d.get('pool_seed', 0))
        self.note = (' (with the test pixels, labels unused)' if d.get('pool_test') else '') + (
            ' (a random subset, seed %d)' % d.get('pool_seed', 0)
            if 0 < d.get('pool_size', 0) == len(self.pool) else '')
        self.sampler = PatchSampler(self.scene, d['patch'], device, d['padding'])

    def describe(self) -> str:
        return ('split %s: train %d, test %d, pool %d%s; train per class %s'
                % (Path(self.split['path']).name, len(self.split['train']), len(self.split['test']), len(self.pool),
                   self.note, class_counts(self.scene, self.split['train']).tolist()))


def write_config(out: Path, cfg: dict):
    (Path(out) / 'ckpt').mkdir(parents=True, exist_ok=True)
    with open(Path(out) / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_one(cfg: dict, st: Setting, k: int, seed: int, device, out: Path, name: str, extra: dict = None,
            log=print) -> dict:
    """Train run k of a setting with `seed` and save it under `out`; returns the record of run<k>.json."""
    out = Path(out)
    split = st.split
    frac = cfg['data'].get('train_fraction', 1.0)
    if frac < 1:                                         # fewer labels: every run draws its own part with its seed
        split = dict(st.split, train=subset_train(st.scene, st.split['train'], frac, seed))
        log('   run %d: %d of the %d training pixels (train_fraction %g); per class %s'
            % (k, len(split['train']), len(st.split['train']), frac, class_counts(st.scene, split['train']).tolist()))
    res = train_run(cfg, st.scene, st.sampler, split, st.pool, seed, device, log=log)
    res['n_train'] = int(len(split['train']))
    torch.save(res.pop('state'), out / 'ckpt' / ('run%d_best.pt' % k))
    torch.save(res.pop('last_state'), out / 'ckpt' / ('run%d_last.pt' % k))
    res.update(dataset=st.dataset, setting=name, run=k, split=st.split['path'], classes=st.scene.classes)
    res.update(extra or {})
    _atomic_json(out / ('run%d.json' % k), res)
    log('   run %d (seed %d): best OA %.2f at epoch %d, last OA %.2f, AA %.2f, %.1f min'
        % (k, res['seed'], res['best']['OA'], res['best_epoch'], res['last']['OA'], res['best']['AA'],
           res['seconds'] / 60))
    return res


def read_run(out: Path, k: int):
    """The record of a finished run k (run<k>.json and both checkpoints on disk), else None."""
    out = Path(out)
    path = out / ('run%d.json' % k)
    if not (path.exists() and (out / 'ckpt' / ('run%d_best.pt' % k)).exists()
            and (out / 'ckpt' / ('run%d_last.pt' % k)).exists()):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            res = json.load(f)
    except (OSError, ValueError):
        return None
    return res if 'best' in res and 'last' in res else None


# --------------------------------------------------------------------------------------------------- summary
def mean_std(values):
    return float(np.mean(values)), float(np.std(values))        # population std, as in Geo-DeltaSSM's tables


def summarise(runs: list) -> dict:
    s = dict(runs=len(runs), best_epoch=float(np.mean([r['best_epoch'] for r in runs])),
             minutes_per_run=float(np.mean([r['seconds'] for r in runs])) / 60)
    for which in ('best', 'last'):
        s[which] = {k: mean_std([r[which][k] for r in runs]) for k in SCORES}
        pc = np.array([r[which]['per_class'] for r in runs])
        s[which]['per_class'] = [mean_std(pc[:, c]) for c in range(pc.shape[1])]
    return s


def write_summary(out: Path, runs: list, classes: list) -> dict:
    s = summarise(runs)
    s.update(parameters=runs[0]['parameters'], classes=classes)
    _atomic_json(Path(out) / 'summary.json', s)
    return s


def table(rows) -> str:
    head = '%-12s %-24s %4s  %-13s %6s %6s %6s  %-13s %6s %7s' % (
        'dataset', 'setting', 'runs', 'OA best', 'AA', 'Kappa', 'F1', 'OA last', 'epoch', 'min/run')
    lines = [head, '-' * len(head)]
    for d, name, s in rows:
        b, l = s['best'], s['last']
        lines.append('%-12s %-24s %4d  %6.2f ± %-4.2f %6.2f %6.2f %6.2f  %6.2f ± %-4.2f %6.0f %7.1f' % (
            d, name, s['runs'], b['OA'][0], b['OA'][1], b['AA'][0], b['Kappa'][0], b['F1'][0], l['OA'][0],
            l['OA'][1], s['best_epoch'], s['minutes_per_run']))
    return '\n'.join(lines)
