"""Evaluate saved SymbioNet checkpoints on clean and corrupted HSI/LiDAR inputs."""
import argparse
import json
import os
from pathlib import Path
import re

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from symbionet.config import load_yaml
from symbionet.conditions import SUITES, suite
from symbionet.runs import find_data_root
from symbionet.testing import Bench, Member, label_of

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path, help='directory containing config.yaml, run*.json and ckpt/')
    parser.add_argument('--weights', choices=('best', 'last'), default='best')
    parser.add_argument('--suite', nargs='+', choices=('clean',) + SUITES, default=['corruptions'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0, help='seed of test corruptions')
    parser.add_argument('--block', type=int, default=16)
    parser.add_argument('--out', type=Path, help='new JSON output file; default: under robustness/')
    a = parser.parse_args(argv)
    if a.chunk <= 0 or a.block <= 0:
        parser.error('--chunk and --block must be positive')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device(a.device)
    folder = a.run_dir.resolve()
    cfg = load_yaml(folder / 'config.yaml')
    if not cfg:
        raise ValueError('Missing or empty training configuration')
    records = []
    for p in folder.glob('run*.json'):
        match = re.fullmatch(r'run(\d+)\.json', p.name)
        if match:
            rec = json.loads(p.read_text(encoding='utf-8'))
            k = int(match.group(1))
            if rec['run'] != k:
                raise ValueError(f'Run index mismatch: {p}')
            records.append((k, rec))
    records.sort(key=lambda r: r[0])
    if not records:
        raise ValueError('No saved training records found')
    dataset = records[0][1]['dataset']
    if any(r['dataset'] != dataset for _, r in records):
        raise ValueError('All records must refer to one dataset')
    if len({r['seed'] for _, r in records}) != len(records):
        raise ValueError('Duplicate seed in saved training records')
    suite_names = list(dict.fromkeys(a.suite))
    output = a.out or ROOT / 'robustness' / f'{folder.parent.parent.name}_{dataset}_{folder.name}_{a.weights}_{"-".join(suite_names)}.json'
    output = output.resolve()
    if output.suffix.lower() != '.json':
        parser.error('--out must be a JSON file')
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}; specify a new --out')
    if output.is_relative_to(folder):
        raise ValueError('Write evaluation output outside the training directory')
    data_root = find_data_root(cfg, ROOT, a.data_root)
    bench = Bench.open(dataset, cfg['data'], ROOT, data_root, device, chunk=a.chunk,
                       seed=a.seed, block=a.block, subset=cfg['eval'].get('subset', 0))
    members = []
    for k, r in records:
        ckpt = folder / 'ckpt' / f'run{k}_{a.weights}.pt'
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        m = Member(a.weights, int(r['seed']), cfg, ckpt, recorded=float(r[a.weights]['OA']))
        bench.load(m)
        members.append(m)
    conds = [None]
    for name in suite_names:
        if name != 'clean':
            conds.extend(suite(name))
    scores_by_condition = {}
    for cond in conds:
        label = label_of(cond)
        values = bench.score(members, cond)
        if cond is None:
            mismatches = [(m.seed, m.recorded, v['OA']) for m, v in zip(members, values)
                          if abs(m.recorded - v['OA']) > 0.02]
            if mismatches:
                raise RuntimeError(f'Clean OA does not match the training record: {mismatches}')
        scores_by_condition[label] = {str(m.seed): v for m, v in zip(members, values)}
        oa = [v['OA'] for v in values]
        print(f'{label:24s} OA {np.mean(oa):.4f} +/- {np.std(oa):.4f}', flush=True)
    report = dict(**bench.header(), checkpoint=a.weights, suites=suite_names,
                  seeds=[m.seed for m in members], scores=scores_by_condition)
    if 'corruptions' in suite_names:
        labels = [label_of(c) for c in suite('corruptions')]
        per_seed = [float(np.mean([scores_by_condition[k][str(m.seed)]['OA'] for k in labels])) for m in members]
        report['all46'] = dict(per_seed=per_seed, mean=float(np.mean(per_seed)), std=float(np.std(per_seed)))
        print(f'All46 OA {np.mean(per_seed):.4f} +/- {np.std(per_seed):.4f}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f'Evaluation saved to {output}')
    return report


if __name__ == '__main__':
    main()
