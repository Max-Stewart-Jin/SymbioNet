"""Robustness of trained models to test-time corruptions: noise, lost pixels, occluded blocks, lost HSI bands and a
missing modality (see symbionet/corrupt.py for the exact definitions). The levels of a test are nested (what is lost at
a level is also lost at every higher level); the block and band tests are repeated with independent draws and averaged.

Examples:
  python robustness.py results/final/houston2013/base results/final/houston2013/symbionet --names base SymbioNet
  python robustness.py "results/final/*/base" "results/final/*/symbionet" --names base SymbioNet --tests noise,missing

Each argument is a result folder written by train.py (config.yaml, run<k>.json, ckpt/run<k>_best.pt), or a pattern
matching one folder per dataset; --names gives one display name per argument. The best checkpoint of every run is
tested on all the test pixels of its split. Folders of the same dataset must share the split and the preprocessing, and
are tested on exactly the same corrupted images. The clean scores are checked against the training records. Results
go to robustness/<out>/<dataset>.json; a summary (mean OA over the runs) is printed at the end and appended to
robustness/<out>/summary.txt.

--set section.key=value changes a setting of every given model at test time only (the checkpoints must still fit),
e.g. --set model.pre_gate=known tests a model that was trained without the gate with the gate switched on.
"""
import os

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')       # as in train.py: same numerics as training

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from symbionet.corrupt import RANDOM, TESTS, band_stats, conditions, corrupt_scene  # noqa: E402
from symbionet.data import PatchSampler, get_split, load_scene  # noqa: E402
from symbionet.engine import build_model, evaluate  # noqa: E402

try:
    import yaml
except ImportError:                     # pragma: no cover
    sys.exit('PyYAML is needed for the configuration files: pip install pyyaml')

SHARED = ('hsi_norm', 'lidar_norm', 'pca', 'split', 'split_ratio', 'split_seed', 'split_file', 'split_dir')
KEEP = ('OA', 'AA', 'Kappa', 'F1')


# ------------------------------------------------------------------------------------------------- folders
def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else ROOT / p


def read_folder(folder: Path) -> dict:
    """One result folder of train.py: its config and, per run, the seed, the recorded best OA and the checkpoint."""
    if not (folder / 'config.yaml').exists():
        raise SystemExit('not a result folder (no config.yaml): %s' % folder)
    with open(folder / 'config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    runs = []
    for p in folder.iterdir():
        m = re.fullmatch(r'run(\d+)\.json', p.name)
        if not m:
            continue
        with open(p, encoding='utf-8') as f:
            res = json.load(f)
        ckpt = folder / 'ckpt' / ('run%s_best.pt' % m.group(1))
        if not ckpt.exists():
            raise SystemExit('missing checkpoint: %s' % ckpt)
        runs.append(dict(run=int(m.group(1)), seed=res['seed'], best_oa=res['best']['OA'], dataset=res['dataset'],
                         ckpt=ckpt))
    if not runs:
        raise SystemExit('no finished runs (run<k>.json) in %s' % folder)
    runs.sort(key=lambda r: r['run'])
    return dict(folder=folder, cfg=cfg, dataset=runs[0]['dataset'], runs=runs)


def collect(patterns, names) -> dict:
    """{dataset: [entry, ...]} in the order given; patterns are expanded here (PowerShell does not expand them)."""
    if names and len(names) != len(patterns):
        raise SystemExit('--names needs one name per folder argument (%d given, %d needed)'
                         % (len(names), len(patterns)))
    groups = {}
    for i, pat in enumerate(patterns):
        found = sorted(glob.glob(str(resolve(pat)))) or [str(resolve(pat))]
        for f in found:
            e = read_folder(Path(f))
            e['name'] = names[i] if names else '%s/%s' % (e['folder'].parent.parent.name, e['folder'].name)
            group = groups.setdefault(e['dataset'], [])
            if any(o['name'] == e['name'] for o in group):
                raise SystemExit('two folders of %s are both called %r (use --names)' % (e['dataset'], e['name']))
            group.append(e)
    return groups


def check_shared(dataset: str, entries: list) -> dict:
    d0 = entries[0]['cfg']['data']
    for e in entries[1:]:
        diff = [k for k in SHARED if e['cfg']['data'].get(k) != d0.get(k)]
        if diff:
            raise SystemExit('%s: %s and %s differ in %s; models of one dataset must share the split and the '
                             'preprocessing' % (dataset, entries[0]['name'], e['name'], ', '.join(diff)))
    if d0['pca']:
        raise SystemExit('%s: robustness tests need pca = 0 (the bands are corrupted before any PCA)' % dataset)
    return d0


def parse_overrides(items) -> dict:
    """{'model.pre_gate': 'known'} from ['model.pre_gate=known']; the values are read as YAML, as in train.py."""
    out = {}
    for item in items or []:
        key, eq, text = item.partition('=')
        if not eq or key.count('.') != 1:
            raise SystemExit('--set expects section.key=value, got %r' % item)
        out[key] = yaml.safe_load(text)
    return out


def override(cfg: dict, over: dict):
    for dotted, value in over.items():
        section, key = dotted.split('.')
        if not isinstance(cfg.get(section), dict):
            raise SystemExit('--set: the configs have no section %r' % section)
        cfg[section][key] = value


# ------------------------------------------------------------------------------------------------- summary
def summary_text(dataset: str, n_test: int, entries: list, res: dict, conds: list, draws: dict) -> str:
    names = [e['name'] for e in entries]
    w = max(12, max(len(n) for n in names) + 2)

    def mean(name, label):
        return float(np.mean(res[name][label]['OA']))

    lines = ['=== %s: %d test pixels; mean OA over the runs (%%)' % (dataset, n_test),
             '%-*s %5s %8s' % (w + 9, 'model', 'runs', 'clean')]
    for e in entries:
        lines.append('%-*s %5d %8.2f' % (w + 9, e['name'], len(e['runs']), mean(e['name'], 'clean')))
    for test in TESTS:
        cs = [c for c in conds if c.test == test]
        if not cs:
            continue
        levels = TESTS[test]['levels']
        more = ' (mean of %d draws)' % draws[test] if draws[test] > 1 else ''
        lines += ['', '%s: %s%s' % (test, TESTS[test]['unit'], more),
                  '  %-*s' % (w + 7, 'level') + ''.join('%8g' % v for v in levels)]
        for target in TESTS[test]['targets']:
            for j, name in enumerate(names):
                vals = ''.join('%8.2f' % mean(name, c.label) for c in cs if c.target == target)
                lines.append('  %-6s %-*s%s' % (target if j == 0 else '', w, name, vals))
    corrupted = [c.label for c in conds]
    lines.append('')
    lines.append('mean OA over all %d corrupted conditions (drop from clean):' % len(corrupted))
    for name in names:
        avg = float(np.mean([mean(name, k) for k in corrupted]))
        lines.append('  %-*s %6.2f  (%+.2f)' % (w, name, avg, avg - mean(name, 'clean')))
    groups = [(t, m) for t in TESTS for m in TESTS[t]['targets'] if any(c.test == t and c.target == m for c in conds)]
    lines += ['', 'mean OA per test and corrupted modality:',
              '  %-*s' % (w, 'model') + ''.join('%14s' % ('%s %s' % g) for g in groups)]
    for name in names:
        lines.append('  %-*s' % (w, name) + ''.join(
            '%14.2f' % np.mean([mean(name, c.label) for c in conds if (c.test, c.target) == g]) for g in groups))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+', help='result folders of train.py (or patterns, one folder per dataset)')
    p.add_argument('--names', nargs='+', help='display name per folder argument')
    p.add_argument('--tests', default=','.join(TESTS), help='comma-separated subset of: %s' % ', '.join(TESTS))
    p.add_argument('--block', type=int, default=16, help='side of the occluding blocks in pixels (block test)')
    p.add_argument('--seed', type=int, default=0, help='seed of the corruptions')
    p.add_argument('--draws', type=int, help='independent draws of every random test (default: per test, see '
                                             'TESTS in symbionet/corrupt.py)')
    p.add_argument('--chunk', type=int, default=1024, help='test patches per forward pass')
    p.add_argument('--data-root', help='folder with Trento/, Houston13/, MUUFL/ (default: data.root of the configs)')
    p.add_argument('--out', default='default', help='name of the output folder under robustness/')
    p.add_argument('--set', nargs='*', default=[], metavar='SECTION.KEY=VALUE',
                   help='test-time change of a setting of every given model, e.g. model.pre_gate=known')
    p.add_argument('--device', help='cuda, cpu, ... (default: cuda if available)')
    a = p.parse_args(argv)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    conds = conditions([t.strip() for t in a.tests.split(',') if t.strip()])
    draws = {t: (a.draws if a.draws and t in RANDOM else TESTS[t]['draws']) for t in TESTS}
    groups = collect(a.folders, a.names)
    over = parse_overrides(a.set)
    for entries in groups.values():
        for e in entries:
            override(e['cfg'], over)
    out_root = ROOT / 'robustness' / a.out
    out_root.mkdir(parents=True, exist_ok=True)
    print('SymbioNet robustness | %d conditions, %d corrupted images | device %s | torch %s'
          % (len(conds), sum(draws[c.test] for c in conds), device, torch.__version__))

    texts = []
    for dataset, entries in groups.items():
        start = time.time()
        d = check_shared(dataset, entries)
        data_root = Path(a.data_root) if a.data_root else resolve(d['root'])
        scene = load_scene(dataset, data_root, d['hsi_norm'], d['lidar_norm'], 0)
        split = get_split(scene, ROOT / d['split_dir'], d['split'], d['split_ratio'], d['split_seed'], d['split_file'])
        test_idx = torch.from_numpy(split['test']).to(device)
        stats = {m: band_stats(getattr(scene, m), scene.valid) for m in ('hsi', 'lidar')}
        print('\n== %s | %s | split %s, %d test pixels' % (dataset, scene.summary(), Path(split['path']).name,
                                                         len(test_idx)))
        for e in entries:                               # all checkpoints, test-time network only
            e['models'] = []
            for r in e['runs']:
                model = build_model(e['cfg'], scene)
                model.load_state_dict(torch.load(r['ckpt'], map_location='cpu', weights_only=True))
                e['models'].append(model.strip_training_branch().to(device).eval())
            print('   %-20s %d runs, patch %d, %s'
                  % (e['name'], len(e['runs']), e['cfg']['data']['patch'], e['folder']))

        res = {e['name']: {} for e in entries}
        for i, cond in enumerate([None] + conds):
            label = 'clean' if cond is None else cond.label
            n_draws = 1 if cond is None else draws[cond.test]
            got = {e['name']: [] for e in entries}     # [draw][run] -> scores
            for draw in range(n_draws):
                cs = scene if cond is None else corrupt_scene(scene, cond, stats, a.seed, a.block, draw)
                samplers = {}
                for e in entries:
                    key = (e['cfg']['data']['patch'], e['cfg']['data']['padding'])
                    if key not in samplers:
                        samplers[key] = PatchSampler(cs, key[0], device, key[1])
                    got[e['name']].append([evaluate(m, samplers[key], test_idx, scene.n_classes, a.chunk)
                                           for m in e['models']])
                del samplers, cs
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            for e in entries:                           # per run: the mean over the draws
                g = got[e['name']]
                res[e['name']][label] = {k: [float(np.mean([g[d][r][k] for d in range(n_draws)]))
                                             for r in range(len(e['models']))] for k in KEEP}
                if n_draws > 1:
                    res[e['name']][label]['OA_per_draw'] = [[g[d][r]['OA'] for d in range(n_draws)]
                                                            for r in range(len(e['models']))]
            print('   [%2d/%d] %-18s %s' % (i, len(conds), label, '  '.join(
                '%s %.2f' % (e['name'], np.mean(res[e['name']][label]['OA'])) for e in entries)))

        # the clean scores must reproduce the training records (same checkpoints, same test pixels)
        bad = []
        for e in entries:
            if e['cfg']['eval'].get('subset'):
                continue                                # trained with a test subset (quick check): no record
            for r, oa in zip(e['runs'], res[e['name']]['clean']['OA']):
                if abs(oa - r['best_oa']) > 0.02:
                    bad.append('%s run %d: %.3f here, %.3f in training' % (e['name'], r['run'], oa, r['best_oa']))
        print('   clean OA ' + ('matches the training records' if not bad else 'DIFFERS: ' + '; '.join(bad)))

        record = dict(dataset=dataset, n_test=len(test_idx), split=split['path'], seed=a.seed, block=a.block,
                      draws=draws,
                      created=time.strftime('%Y-%m-%d %H:%M'), torch=torch.__version__,
                      models=[dict(name=e['name'], folder=str(e['folder']), runs=[r['run'] for r in e['runs']],
                                   seeds=[r['seed'] for r in e['runs']], **({'set': over} if over else {}))
                              for e in entries],
                      conditions=['clean'] + [c.label for c in conds], scores=res, clean_check=bad)
        with open(out_root / ('%s.json' % dataset), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=1)
        texts.append(summary_text(dataset, len(test_idx), entries, res, conds, draws))
        print('   %.1f min' % ((time.time() - start) / 60))

    stamp = time.strftime('%Y-%m-%d %H:%M')
    text = '\n\n'.join(texts)
    with open(out_root / 'summary.txt', 'a', encoding='utf-8') as f:
        f.write('%s  python robustness.py %s\n%s\n\n' % (stamp, ' '.join(sys.argv[1:] if argv is None else argv), text))
    print('\n' + '=' * 30 + ' SymbioNet robustness (%s, %s) ' % (a.out, stamp) + '=' * 30)
    print(text)
    print('Details: %s' % out_root)


if __name__ == '__main__':
    main()
