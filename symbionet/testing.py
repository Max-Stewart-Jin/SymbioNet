"""Scores of trained models on the test pixels of a scene under test conditions, and the file they are kept in.

A Bench holds one dataset: the clean scene, the split and its test pixels. Bench.score() makes the test scene of a
condition once (symbionet/conditions.py) and scores every given model on it, so all models see exactly the same
images; a condition with several draws is scored on each draw and averaged per model. A Store keeps the scores of
one dataset in one JSON file, by model, condition and seed, and is written after every condition: a test stage can
be stopped at any time, and only what is missing is computed the next time.

"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .conditions import apply, n_draws
from .corrupt import band_stats
from .data import PatchSampler, get_split, load_scene
from .engine import build_model, predict
from .metrics import confusion, scores

KEEP = ('OA', 'AA', 'Kappa', 'F1')
CLEAN = 'clean'
SHARED = ('hsi_norm', 'lidar_norm', 'pca', 'split', 'split_ratio', 'split_seed', 'split_file', 'split_dir')


@dataclass
class Member:
    """One trained run that is tested: `key` names its model (e.g. 'full/best'), `seed` the run."""
    key: str
    seed: int
    cfg: dict
    ckpt: Path
    recorded: float = None                                          # the clean OA of the training record, if any
    model: object = None


def label_of(cond) -> str:
    return CLEAN if cond is None else cond.label


class Bench:
    def __init__(self, scene, split: dict, data: dict, device, chunk: int = 1024, seed: int = 0, block: int = 16,
                 subset: int = 0):
        """scene, split: the clean scene and its split (symbionet/data.py); data: the data section of the settings
        of the models, which must share the entries in SHARED. seed and block: of the random conditions. subset > 0
        tests a fixed random part of the test pixels, the same part that training uses with eval.subset."""
        if data.get('pca'):
            raise ValueError('%s: test conditions need pca = 0 (the bands are changed before any PCA)' % scene.name)
        self.dataset, self.device, self.chunk, self.seed, self.block = scene.name, device, chunk, seed, block
        self.data = {k: data.get(k) for k in SHARED}
        self.scene, self.split = scene, split
        test = self.split['test']
        if subset:
            keep = np.random.default_rng(0).choice(len(test), min(subset, len(test)), replace=False)
            test = test[np.sort(keep)]
        self.test = test
        self.test_idx = torch.from_numpy(test).to(device)
        self.y = self.scene.labels.ravel()[test]
        self.stats = {m: band_stats(getattr(self.scene, m), self.scene.valid) for m in ('hsi', 'lidar')}

    @classmethod
    def open(cls, dataset: str, data: dict, root, data_root, device, **kw):
        """The bench of a dataset on disk: root holds the splits (data.split_dir), data_root the images."""
        scene = load_scene(dataset, data_root, data['hsi_norm'], data['lidar_norm'], 0)
        split = get_split(scene, Path(root) / data['split_dir'], data['split'], data['split_ratio'],
                          data['split_seed'], data['split_file'])
        return cls(scene, split, data, device, **kw)

    def header(self) -> dict:
        """What must be equal for two sets of scores to share a file."""
        return dict(dataset=self.dataset, n_test=int(len(self.test)), split=Path(self.split['path']).name,
                    seed=self.seed, block=self.block)

    def fits(self, member: Member) -> list:
        """The shared data settings in which a member differs from this bench (empty: it can be tested here)."""
        return [k for k in SHARED if member.cfg['data'].get(k) != self.data[k]]

    def load(self, member: Member):
        """The test-time network of a member, on the device."""
        model = build_model(member.cfg, self.scene)
        model.load_state_dict(torch.load(member.ckpt, map_location='cpu', weights_only=True))
        member.model = model.strip_training_branch().to(self.device).eval()
        return member.model

    def score(self, members: list, cond=None, on_clean=None) -> list:
        """The scores of every member under one condition (None: the clean scene), in the order of `members`:
        {'OA', 'AA', 'Kappa', 'F1'} as the mean over the draws of the condition, plus 'OA_per_draw' when there are
        several. on_clean(member, predictions) is called with the predictions on the clean scene (a numpy array,
        one class per test pixel)."""
        draws = 1 if cond is None else n_draws(cond)
        got = [[] for _ in members]
        for draw in range(draws):
            view = self.scene if cond is None else apply(self.scene, cond, self.stats, self.seed, self.block, draw)
            samplers = {}
            for i, m in enumerate(members):
                key = (m.cfg['data']['patch'], m.cfg['data']['padding'])
                if key not in samplers:
                    samplers[key] = PatchSampler(view, key[0], self.device, key[1])
                pred = predict(m.model, samplers[key], self.test_idx, self.chunk)
                got[i].append(scores(confusion(pred, samplers[key].targets(self.test_idx), self.scene.n_classes)))
                if cond is None and on_clean is not None:
                    on_clean(m, pred.cpu().numpy())
            del samplers, view
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        out = []
        for g in got:
            s = {k: float(np.mean([d[k] for d in g])) for k in KEEP}
            if draws > 1:
                s['OA_per_draw'] = [float(d['OA']) for d in g]
            out.append(s)
        return out


# ------------------------------------------------------------------------------------------------------ store
class Store:
    """The test scores of one dataset: scores[key][label][seed] = {'OA': ..., ...}."""

    def __init__(self, path, header: dict):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, encoding='utf-8') as f:
                self.data = json.load(f)
            diff = [k for k, v in header.items() if self.data.get(k) != v]
            if diff:
                raise ValueError('%s holds scores of another test setting (%s differ); move it away to start again'
                                 % (self.path, ', '.join(diff)))
        else:
            self.data = dict(header, created=time.strftime('%Y-%m-%d %H:%M'), models={}, scores={})

    @classmethod
    def read(cls, path):
        """The stored scores without a header to check against, or None if there is no file."""
        path = Path(path)
        if not path.exists():
            return None
        store = cls.__new__(cls)
        store.path = path
        with open(path, encoding='utf-8') as f:
            store.data = json.load(f)
        return store

    def has(self, key: str, label: str, seed: int) -> bool:
        return str(seed) in self.data['scores'].get(key, {}).get(label, {})

    def get(self, key: str, label: str) -> dict:
        """{seed: scores} of a model under one condition (empty if nothing is stored)."""
        return {int(s): v for s, v in self.data['scores'].get(key, {}).get(label, {}).items()}

    def put(self, key: str, label: str, seed: int, value: dict):
        self.data['scores'].setdefault(key, {}).setdefault(label, {})[str(seed)] = value

    def describe(self, key: str, **info):
        self.data['models'].setdefault(key, {}).update(info)

    def save(self):
        self.data['updated'] = time.strftime('%Y-%m-%d %H:%M')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + '.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=1)
        os.replace(tmp, self.path)


def mean_over_seeds(store: Store, key: str, labels, metric: str = 'OA', seeds=None):
    """The mean over the given conditions of the mean over the seeds, using only the seeds that have every one of
    the conditions (None: all such seeds). Returns (mean, per-seed values, seeds), or (None, [], []) when no seed
    is complete."""
    labels = list(labels)
    per = [store.get(key, lb) for lb in labels]
    common = set(per[0]) if per else set()
    for p in per[1:]:
        common &= set(p)
    if seeds is not None:
        common &= set(seeds)
    common = sorted(common)
    if not common:
        return None, [], []
    values = [float(np.mean([p[s][metric] for p in per])) for s in common]
    return float(np.mean(values)), values, common
