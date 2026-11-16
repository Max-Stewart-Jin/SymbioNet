"""Train and test CNN_H with the labeled masked copy used by the copies ablation."""

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import closing
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch

from scripts import compare
import formal
from scripts.cnn_copies_eval import evaluate, summarize
from scripts.cnn_copies_train import train
from symbionet import external
from symbionet.config import load_yaml
from symbionet.data import split_path
from symbionet.runs import Setting, _atomic_json, find_data_root, read_run, write_config, write_summary

VERSION = 'ext_cnnh_copies'
ROOT = Path(__file__).resolve().parents[1]
FORMAL_SPLITS = {
    'houston2013': 'd852b79976f44b9574624a401912b7151fe889fff816eb41e7645caec30127ce',
    'muufl': '05b844ff93f77264dbb9209c198c9173c7b31fd6c59756b28c4dfeda31a6ffab',
    'trento': 'f021439a5503d21beb96134972d07c440a5df2562334e29cab42be4893448fdc',
}


def split_digest(split):
    """Fingerprint ordered train/test indices, independent of the NPZ encoding."""
    h = hashlib.sha256()
    for key in ('train', 'test'):
        indices = np.asarray(split[key], dtype='<i8').ravel()
        h.update(len(indices).to_bytes(8, 'little'))
        h.update(indices.tobytes())
    return h.hexdigest()


def check_references(plan, datasets, seeds, source_digest):
    """Check the actual CNN_H source and settings against the records we intend to pair."""
    for dataset in datasets:
        baseline = plan.folder(dataset, 'ext_cnnh')
        recorded = load_yaml(baseline / 'config.yaml')
        recorded.pop('compared')
        differences = formal.same_settings(recorded, plan.config(dataset, 'base'))
        recipe = plan.config(dataset, 'copies')
        data = recipe['data']
        indices_path = data['split_file'] or split_path(plan.root / data['split_dir'], dataset, data['split'],
                                                       data['split_ratio'], data['split_seed'])
        with np.load(indices_path) as split:
            if split_digest(split) != FORMAL_SPLITS[dataset]:
                raise ValueError('train/test pixels differ from the formal split: %s' % indices_path)
        differences += [key for key in ('data', 'training', 'eval') if recipe[key] != recorded[key]]
        differences += formal.same_settings(recipe, load_yaml(plan.folder(dataset, 'copies') / 'config.yaml'))
        if differences:
            raise ValueError('%s: settings differ from the formal runs: %s' % (dataset, ', '.join(differences)))
        for seed in seeds:
            path = baseline / ('run%d.json' % seed)
            with path.open(encoding='utf-8') as stream:
                run = json.load(stream)
            if (run['seed'], run['dataset'], run['setting']) != (seed, dataset, 'ext_cnnh'):
                raise ValueError('unexpected reference run in %s' % path)
            if run['compared']['sha256'] != source_digest:
                raise ValueError('CNN_H source differs from the one used in %s' % path)


def save_weights(path, weights):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(weights, temporary)
    os.replace(temporary, path)


def train_datasets(plan, output, datasets, seeds, ext, data_root, device, source_digest, say):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device)
    sources = formal.code_id(plan.root)
    for name in ('scripts/cnn_copies.py', 'scripts/cnn_copies_train.py', 'scripts/cnn_copies_eval.py',
                 'symbionet/external.py',
                 'symbionet/testing.py', 'symbionet/conditions.py'):
        sources[name] = external.source_hash(plan.root / name)[:10]
    for dataset in datasets:
        cfg = plan.config(dataset, 'copies')
        out = output / dataset / VERSION
        config = dict(cfg, compared=dict(method='cnnh', protocol='copies', source_sha256=source_digest))
        if (out / 'config.yaml').is_file():
            differences = formal.same_settings(load_yaml(out / 'config.yaml'), config)
            if differences:
                raise ValueError('stored settings differ in %s: %s' % (out, ', '.join(differences)))
        else:
            write_config(out, config)

        setting = None
        records = []
        for seed in seeds:
            record = read_run(out, seed)
            if record is not None:
                if (record['setting'], record['seed'], record['dataset']) != (VERSION, seed, dataset):
                    raise ValueError('unexpected run in %s/run%d.json' % (out, seed))
                records.append(record)
                say('%s seed %d: training already complete' % (dataset, seed))
                continue
            if setting is None:
                setting = Setting(cfg, dataset, plan.root, find_data_root(cfg, plan.root, data_root), device)
                say(setting.describe())
            say('%s seed %d: training %s' % (dataset, seed, VERSION))
            record = train(cfg, setting.scene, setting.sampler, setting.split, seed, device, ext, log=say)
            save_weights(out / 'ckpt' / ('run%d_best.pt' % seed), record.pop('state'))
            save_weights(out / 'ckpt' / ('run%d_last.pt' % seed), record.pop('last_state'))
            record.update(dataset=dataset, setting=VERSION, run=seed, split=setting.split['path'],
                          split_sha256=split_digest(setting.split),
                          n_train=len(setting.split['train']), classes=setting.scene.classes, device=device_name,
                          torch=torch.__version__, protocol='copies', finished=time.strftime('%Y-%m-%d %H:%M'),
                          code=sources)
            _atomic_json(out / ('run%d.json' % seed), record)
            records.append(record)
            say('%s seed %d: best clean OA %.2f, epoch %d' % (dataset, seed, record['best']['OA'],
                                                              record['best_epoch']))
        write_summary(out, records, records[0]['classes'])


def pack_results(output, datasets, seeds, summary):
    path = output.parent / 'SymbioNet_cnn_copies_results.zip'
    files = [summary, output / 'log.txt']
    for dataset in datasets:
        folder = output / dataset / VERSION
        files += [folder / 'config.yaml', folder / 'summary.json', output / 'tests' / (dataset + '.json')]
        files += [folder / ('run%d.json' % seed) for seed in seeds]
    temporary = path.with_suffix('.zip.tmp')
    with ZipFile(temporary, 'w', ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, file.relative_to(output.parent))
    os.replace(temporary, path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'smoke', 'all'))
    parser.add_argument('--external', type=Path, default=ROOT / 'external')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--device', help='cpu, cuda, cuda:0, ...')
    parser.add_argument('--datasets', help='smoke only; comma-separated, default houston2013')
    parser.add_argument('--seeds', help='smoke only; comma-separated, default 0')
    args = parser.parse_args()
    smoke = args.command == 'smoke'
    if not smoke and (args.datasets or args.seeds):
        parser.error('--datasets and --seeds apply to smoke only')
    plan = formal.Plan(smoke=smoke)
    datasets = [d.strip() for d in (args.datasets or 'houston2013').split(',')] if smoke else list(plan.raw['datasets'])
    seeds = [int(s) for s in (args.seeds or '0').split(',')] if smoke else plan.seeds
    if (len(datasets) != len(set(datasets)) or len(seeds) != len(set(seeds))
            or not set(datasets) <= set(plan.raw['datasets']) or not set(seeds) <= set(plan.seeds)):
        parser.error('use datasets and seed numbers from configs/formal.yaml once each')
    path, warning = external.locate('cnnh', args.external)
    digest = external.source_hash(path)
    print('CNN_H source: %s (%s)' % (path, digest), flush=True)
    if warning:
        print(warning, flush=True)
    if not smoke:
        check_references(plan, datasets, seeds, digest)
    if args.command == 'check':
        print('CNN_H source, train/test indices, formal settings and %d reference run records match.'
              % (len(datasets) * len(seeds)))
        return 0

    output = ROOT / 'results' / ('cnn_copies_smoke' if smoke else 'cnn_copies')
    # The original launcher holds launcher.lock across all its stages. Use its queue even for a smoke run.
    queue = formal.Plan()
    print('Waiting for the formal experiment queue. Press Ctrl+C to cancel.', flush=True)
    lock = compare.take_lock(queue, wait=True, quiet=0 if smoke else compare.QUIET)
    if not lock:
        raise RuntimeError('could not acquire the formal experiment lock at %s' % queue.out_root)
    with closing(lock):
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        say = formal.Tee(output / 'log.txt')
        formal.keep_awake(True)
        try:
            say('%s | %s | %s | %s' % (time.strftime('%Y-%m-%d %H:%M'), args.command, device, output))
            train_datasets(plan, output, datasets, seeds, args.external, args.data_root, device, digest, say)
            evaluate(plan, output, datasets, seeds, args.external, args.data_root, device, say=say)
            if smoke:
                say('Smoke training and scoring complete: %s' % output)
            else:
                summary = summarize(plan.out_root, output, datasets, seeds, say=say)
                say('Finished all paired comparisons.')
                say('Results package: %s' % pack_results(output, datasets, seeds, summary))
        finally:
            formal.keep_awake(False)
            say.f.close()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('Stopped. Run the same command to continue.')
