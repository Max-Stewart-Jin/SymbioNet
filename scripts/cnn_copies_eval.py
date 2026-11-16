"""Score the CNN_H transfer and compare paired gains against the original formal runs."""
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev, stdev

from symbionet.conditions import suite


VERSION = 'ext_cnnh_copies'
KEY = VERSION + '/best'
METRICS = ('OA', 'AA', 'Kappa', 'F1')
HEADERS = ('dataset', 'n_test', 'split', 'seed', 'block')
FORMAL_DATASETS = ('houston2013', 'muufl', 'trento')
FORMAL_SEEDS = (0, 1, 2, 3, 4)


def _record(folder, dataset, version, seed):
    path = folder / ('run%d.json' % seed)
    with path.open(encoding='utf-8') as f:
        record = json.load(f)
    if (record['dataset'], record['setting'], record['seed'], record['run']) != (dataset, version, seed, seed):
        raise ValueError('unexpected run identity in %s' % path)
    return record


def _checkpoint_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _check_clean(scores, record, dataset, version, seed):
    for metric in METRICS:
        actual, expected = scores[metric], record['best'][metric]
        if not math.isfinite(actual) or not math.isfinite(expected) or abs(actual - expected) > 0.02:
            raise ValueError('%s/%s seed %d: clean %s %.4f, training record %.4f'
                             % (dataset, version, seed, metric, actual, expected))


def evaluate(plan, output_root, datasets, seeds, external_root, data_root, device, say=print):
    """Resume clean and corruption scoring in the candidate tree. The caller holds the experiment lock."""
    import torch
    from scripts.compare import conditions
    from symbionet import external
    from symbionet.config import load_yaml
    from symbionet.runs import find_data_root
    from symbionet.testing import Bench, Member, Store, label_of

    device = torch.device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output_root = Path(output_root)
    for dataset in datasets:
        folder = output_root / dataset / VERSION
        cfg = load_yaml(folder / 'config.yaml')
        records = {s: _record(folder, dataset, VERSION, s) for s in seeds}
        members = [Member(KEY, s, cfg, folder / 'ckpt' / ('run%d_best.pt' % s), r['best']['OA'])
                   for s, r in records.items()]
        bench = Bench.open(dataset, cfg['data'], plan.root, find_data_root(cfg, plan.root, data_root), device,
                           chunk=cfg['eval']['chunk'], subset=cfg['eval']['subset'])
        store = Store(output_root / 'tests' / (dataset + '.json'), bench.header())
        store.describe(KEY, version=VERSION, weights='best', method='CNN_H + masked labeled copies')
        hashes = store.data['models'][KEY].setdefault('checkpoint_sha256', {})
        for m in members:
            seed_key = str(m.seed)
            digest = _checkpoint_digest(m.ckpt)
            if hashes.get(seed_key) != digest:
                for scores_by_seed in store.data['scores'].get(KEY, {}).values():
                    scores_by_seed.pop(seed_key, None)
                hashes[seed_key] = digest

        # Persist invalidation before any scoring, so a retry cannot reuse an old checkpoint's scores.
        store.save()
        work = []
        for cond in conditions(plan):
            selected = [m for m in members if not store.has(KEY, label_of(cond), m.seed)]
            if selected:
                work.append((cond, selected))
        for m in members:
            if store.has(KEY, 'clean', m.seed):
                _check_clean(store.get(KEY, 'clean')[m.seed], records[m.seed], dataset, VERSION, m.seed)
        needed = {m.seed for _, selected in work for m in selected}
        for m in members:
            if m.seed in needed:
                model = external.build('cnnh', bench.scene, cfg['data']['patch'], external_root)
                model.load_state_dict(torch.load(m.ckpt, map_location='cpu', weights_only=True))
                m.model = model.to(device).eval()
        say('%s: %d test pixels, %d checkpoint(s) with pending scores' % (dataset, len(bench.test), len(needed)))
        for i, (cond, selected) in enumerate(work, 1):
            for m, result in zip(selected, bench.score(selected, cond)):
                if cond is None:
                    _check_clean(result, records[m.seed], dataset, VERSION, m.seed)
                store.put(KEY, label_of(cond), m.seed, result)
            store.save()
            say('  [%d/%d] %s (%d seeds)' % (i, len(work), label_of(cond), len(selected)))
        for m in members:
            m.model = None
        del bench
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def _seed_scores(store, record, version, labels):
    dataset, seed = record['dataset'], record['seed']
    scores = store['scores'][version + '/best']
    missing = [lb for lb in ['clean', *labels] if str(seed) not in scores.get(lb, {})]
    if missing:
        raise ValueError('incomplete scores for %s/%s seed %d: %s' % (dataset, version, seed, ', '.join(missing)))
    clean = scores['clean'][str(seed)]
    _check_clean(clean, record, dataset, version, seed)
    damaged = [scores[label][str(seed)]['OA'] for label in labels]
    if not all(math.isfinite(v) for v in damaged):
        raise ValueError('non-finite damaged OA for %s/%s seed %d' % (dataset, version, seed))
    return {**{'clean_' + k: clean[k] for k in METRICS}, 'damaged_OA': mean(damaged)}


def _delta(after, before):
    return {k: v - before[k] for k, v in after.items()}


def _stats(rows):
    columns = {k: [row[k] for row in rows] for k in rows[0]}
    return {k: {'mean': mean(values), 'sd_population': pstdev(values),
                'sd_sample': stdev(values) if len(values) > 1 else None}
            for k, values in columns.items()}


def summarize(reference_root, output_root, datasets, seeds, say=print):
    """Read both score trees. Write the comparison only when every requested seed has all 46 conditions."""
    reference_root, output_root = Path(reference_root), Path(output_root)
    datasets, seeds = list(datasets), list(seeds)
    if (not datasets or not seeds or len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds)
            or not set(datasets) <= set(FORMAL_DATASETS) or not set(seeds) <= set(FORMAL_SEEDS)):
        raise ValueError('select unique datasets and seeds from the formal plan')
    labels = [c.label for c in suite('corruptions')]
    if len(set(labels)) != 46 or len(labels) != 46:
        raise ValueError('the paired report requires the 46 distinct formal corruptions')
    formal = set(datasets) == set(FORMAL_DATASETS) and set(seeds) == set(FORMAL_SEEDS)
    report = {'complete_formal_selection': formal, 'seeds': seeds, 'conditions': labels,
              'units': 'scores in percent, gains in percentage points, time in seconds',
              'weights': 'best clean test OA checkpoint, following the original comparison',
              'aggregation': 'Mean OA across the 46 conditions within each seed. Paired gains and their difference '
                             'are computed within seed, then averaged across seeds. sd_population uses n, as in '
                             'the formal tables; sd_sample uses n-1 and is undefined for a single seed.',
              'gain_difference': '(copies - base) - (ext_cnnh_copies - ext_cnnh)', 'datasets': {}}
    versions = ('ext_cnnh', VERSION, 'base', 'copies')
    for dataset in datasets:
        with (reference_root / 'tests' / (dataset + '.json')).open(encoding='utf-8') as f:
            original = json.load(f)
        with (output_root / 'tests' / (dataset + '.json')).open(encoding='utf-8') as f:
            candidate = json.load(f)
        header = {k: original[k] for k in HEADERS}
        if header['dataset'] != dataset or any(candidate[k] != v for k, v in header.items()):
            raise ValueError('test headers differ for %s: check split, test size, corruption seed and block size'
                             % dataset)
        rows = []
        for s in seeds:
            absolutes, seconds, devices, counts = {}, {}, {}, []
            for v in versions:
                is_new = v == VERSION
                root, store = (output_root, candidate) if is_new else (reference_root, original)
                record = _record(root / dataset / v, dataset, v, s)
                if str(record['split']).replace('\\', '/').split('/')[-1] != header['split']:
                    raise ValueError('split mismatch for %s/%s seed %d' % (dataset, v, s))
                counts.append(record['n_train'])
                absolutes[v] = _seed_scores(store, record, v, labels)
                seconds[v] = record['seconds']
                devices[v] = record.get('device')
            if len(set(counts)) != 1:
                raise ValueError('different training set sizes on %s seed %d: %s' % (dataset, s, counts))
            cnnh = _delta(absolutes[VERSION], absolutes['ext_cnnh'])
            symbio = _delta(absolutes['copies'], absolutes['base'])
            rows.append(dict(seed=s, scores=absolutes, cnnh_gain=cnnh, symbio_gain=symbio,
                             gain_difference=_delta(symbio, cnnh), training_seconds=seconds, devices=devices))

        agg = {name: _stats([r[name] for r in rows]) for name in
               ('cnnh_gain', 'symbio_gain', 'gain_difference', 'training_seconds')}
        agg['scores'] = {v: _stats([r['scores'][v] for r in rows]) for v in versions}
        report['datasets'][dataset] = {'test': header, 'per_seed': rows, 'summary': agg}
        say('%s (%d paired seeds) damaged OA gains: CNN_H %+.3f, SymbioNet backbone %+.3f, difference %+.3f pp'
            % (dataset, len(seeds), agg['cnnh_gain']['damaged_OA']['mean'],
               agg['symbio_gain']['damaged_OA']['mean'], agg['gain_difference']['damaged_OA']['mean']))

    name = 'paired_summary' if formal else 'paired_%s_seeds%s' % ('_'.join(datasets), '-'.join(map(str, seeds)))
    path = output_root / (name + '.json')
    tmp = path.with_suffix('.json.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    os.replace(tmp, path)
    return path
