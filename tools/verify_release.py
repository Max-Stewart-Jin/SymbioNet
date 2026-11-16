#!/usr/bin/env python3
"""Check the public repository without training models or accessing private data.

Usage: python tools/verify_release.py

This is a source-integrity and static-data check, NOT a full numerical reproduction.
"""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'ORIGINAL_SNAPSHOT_SHA256.json'


def main():
    expected = json.loads(MANIFEST.read_text(encoding='utf-8'))['files']
    errors = []
    for rel, digest in expected.items():
        path = ROOT / rel
        if not path.is_file():
            errors.append(f'MISSING: {rel}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f'CHANGED SINCE ORIGINAL SNAPSHOT: {rel}')
    if errors:
        for error in errors:
            print('ERROR:', error)
        print('Historical source comparison FAILED; review any intended changes against the paper.')
        return 1

    # Compile sources in memory. Never writes __pycache__ files or executes training.
    modules = sorted(ROOT.rglob('*.py'))
    for module in modules:
        if any(part in {'.venv', 'venv', 'external', 'results'} for part in module.relative_to(ROOT).parts):
            continue
        try:
            compile(module.read_bytes(), str(module), 'exec')
        except (SyntaxError, UnicodeError) as err:
            print(f'ERROR: Python syntax in {module.relative_to(ROOT)}: {err}')
            return 1

    try:
        import numpy as np
        import yaml
    except ImportError as exc:
        print(f'ERROR: install requirements.txt first ({exc})')
        return 1

    plan = yaml.safe_load((ROOT / 'configs' / 'formal.yaml').read_text(encoding='utf-8'))
    if plan.get('seeds') != [0, 1, 2, 3, 4]:
        print('ERROR: unexpected formal experiment seed list')
        return 1
    names = ('houston2013', 'muufl', 'trento')
    expected_counts = {'houston2013': (158, 14871), 'muufl': (541, 53146), 'trento': (306, 29908)}
    for name in names:
        path = ROOT / 'splits' / f'{name}_geodeltassm.npz'
        if not path.is_file():
            print(f'ERROR: missing fixed split {path.name}')
            return 1
        try:
            with np.load(path, allow_pickle=False) as arrays:
                train = np.asarray(arrays['train']).reshape(-1)
                test = np.asarray(arrays['test']).reshape(-1)
                shape = tuple(int(n) for n in np.asarray(arrays['shape']).reshape(-1))
        except (KeyError, OSError, ValueError) as exc:
            print(f'ERROR: cannot read {path.name}: {exc}')
            return 1
        if train.ndim != 1 or test.ndim != 1 or (len(train), len(test)) != expected_counts[name]:
            print(f'ERROR: unexpected number of train/test indices for {name}')
            return 1
        if not shape or len(shape) != 2 or any(n <= 0 for n in shape):
            print(f'ERROR: invalid scene shape for {name}')
            return 1
        if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(test.dtype, np.integer):
            print(f'ERROR: non-integer split indices for {name}')
            return 1
        if (np.any(train < 0) or np.any(test < 0) or
            np.any(train >= shape[0] * shape[1]) or np.any(test >= shape[0] * shape[1]) or
            len(np.unique(train)) != len(train) or len(np.unique(test)) != len(test) or
            np.intersect1d(train, test).size):
            print(f'ERROR: invalid or overlapping split indices for {name}')
            return 1
        print(f'OK {name}: {len(train)} train / {len(test)} test; fixed split indices valid')

    print(f'OK original snapshot: {len(expected)} files with matching SHA-256')
    print(f'OK Python syntax: {len(modules)} module(s) checked (no training)')
    print('PASS: source and split validation only; this is NOT a full GPU or paper reproduction.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
