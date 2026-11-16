#!/usr/bin/env python3
"""How does the M2C branch complete a masked token? Two tests on the trained models, without any training.

    python completion_check.py                     (reads the trained models of results/formal, writes nothing there)
    python completion_check.py --patches 500 --seeds 0,1 --datasets muufl     (a quicker look)

For every model of the version (default: full, the method of the paper, weights of the reported epoch) and a fixed
sample of patches around test pixels, the masks are drawn as in training (complementary: at every masked position the
other modality is visible) and the heads f_H, f_L predict the tokens of the unmasked patch, the targets of L_feat.
The error is that of L_feat: the squared difference between the predicted and the clean token, averaged over the
feature dimension, then over the masked positions of a group.

Test 1, the same position: is the value of the other modality at the SAME position used?
  A  the training condition.
  B  the same masks, and for a random half of the masked positions of each modality the other modality is hidden as
     well (it gets the fill value, exactly like a masked value). The masked positions of a modality then fall into
     two groups inside the same forward pass:
        visible   the other modality is still there at that position
        hidden    the other modality was hidden there too
  ratio = error(hidden) / error(visible). Both groups lose the same amount of surrounding context, so the ratio
  isolates the effect of the value at the same position. R2 = 1 - error / (error of predicting the mean token of the
  patch), the share of the variation of the target tokens over the patch that the prediction explains.

Test 2, the whole modality: is the other modality used at all?
  The masks of the own modality stay as drawn. The other modality is
        shown     visible at every position
        trained   masked as in training (condition A: visible exactly where the own modality is masked)
        hidden    hidden at every position
  ratio = error(hidden) / error(trained): how much worse a modality is completed when the part of the other modality
  that training leaves visible is taken away. The ratio is also given by the share of the own modality that is masked
  in the patch (below 1/3, 1/3 to 2/3, 2/3 and above): a modality that is almost complete may not need the other one.

The BatchNorm layers of the stems normalise in one of two ways:
  training  as the masked views in training (bn_masked: batch): the targets with the statistics of the clean batch,
            the masked batch with its own statistics. Condition B and the states of test 2 keep the statistics of the
            batch masked as in training, so an intervention changes the input and not the normalisation. The heads
            were trained under this condition, so its verdicts are the ones for statements about the M2C branch.
  test      with the running statistics, as the deployed network does.
In both, the gate acts on every sample, as in the deployed network (in training it acts on a random half).

The rule, applied to each direction (HSI tokens, LiDAR tokens) and each normalisation separately. It was written
before the first run of each test (test 1 as at test time: 2026-11-13 morning; the normalisation of training and
test 2: 2026-11-13 noon):
  yes                   ratio >= 1.5 on every dataset (mean over the seeds) and > 1 for every single model
  to a limited degree   ratio >= 1.1 on every dataset, but not the line above
  no                    otherwise
Whatever comes out is reported as it is.

The script runs on the CPU by default, so that it does not compete with a training or test stage for the GPU, reads
the checkpoints only, and writes one file outside results/formal: results/analysis/completion_check.json (a file of
an earlier run is kept under a name with its time).
"""
import argparse
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import formal  # noqa: E402
from symbionet.engine import build_model  # noqa: E402
from symbionet.runs import Setting, find_data_root  # noqa: E402

GROUPS = ('train', 'visible', 'hidden')          # test 1: condition A; condition B with / without the other modality
STATES = ('shown', 'train', 'hidden')            # test 2: the other modality (train = condition A)
BINS = ('all', 'low', 'mid', 'high')             # masked share of the own modality: any, < 1/3, < 2/3, the rest
NORMS = {'training': 'normalisation of training (the condition of the M2C branch)',
         'test': 'normalisation as at test time (the deployed network)'}
NAMES = {'h': 'HSI', 'l': 'LiDAR'}
STRONG, WEAK = 1.5, 1.1


def half_of(mask, g):
    """A random half of the masked positions of every row (each masked position with probability 1/2)."""
    return mask * (torch.rand(mask.shape, generator=g) < 0.5).to(mask)


@contextmanager
def batch_statistics(model, active=True):
    """Yields adopt(fn). Active: fn() runs with the BatchNorm layers normalising by the statistics of its batch, as
    the masked views in training (bn_masked='batch'), and exactly these statistics stay in the layers for the passes
    that follow, until the next adopt(). The running statistics of the model are put back at the end. Not active (or
    a model without BatchNorm): adopt(fn) is fn()."""
    bns = [x for x in model.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)] if active else []
    if not bns:
        yield lambda fn: fn()
        return
    saved = [(x.momentum, x.training, {k: v.clone() for k, v in x._buffers.items() if v is not None}) for x in bns]

    def adopt(fn):
        values = {}                                          # values per channel in the batch of every layer
        hooks = [x.register_forward_hook(lambda mod, inp, out: values.__setitem__(mod, inp[0].numel() // inp[0].shape[1]))
                 for x in bns]
        for x in bns:
            x.momentum = 1.0                                 # running statistics := those of this batch
            x.train()
        try:
            return fn()
        finally:
            for x, hook in zip(bns, hooks):
                hook.remove()
                x.eval()
                n = values.get(x, 0)
                if n > 1:                                    # the layer stored the unbiased variance and normalised
                    x.running_var.mul_((n - 1) / n)          # with the biased one: keep the one it normalised with

    try:
        yield adopt
    finally:
        for x, (momentum, training, buffers) in zip(bns, saved):
            x.momentum = momentum
            x.train(training)
            for k, v in buffers.items():
                x._buffers[k].copy_(v)


@torch.no_grad()
def measure(model, sampler, idx, chunk, norm='test'):
    """Sums over the masked positions, per modality: 'position' {group: [error, reference, positions]} (test 1) and
    'modality' {state: {bin: [error, positions]}} (test 2). The masks are the same for every model and normalisation."""
    acc = {m: dict(position={k: np.zeros(3) for k in GROUPS},
                   modality={s: {b: np.zeros(2) for b in BINS} for s in STATES}) for m in 'hl'}
    g = torch.Generator().manual_seed(0)
    with batch_statistics(model, norm == 'training') as adopt:
        for part in idx.split(chunk):
            h, l = sampler(part)
            mh, ml = model.draw_masks(len(part), g, h.device)
            mh, ml = mh.to(h), ml.to(h)
            sh, sl = half_of(mh.cpu(), g).to(h), half_of(ml.cpu(), g).to(h)
            ch, cl = adopt(lambda: model.encode(h, l))                   # the targets: tokens of the unmasked patch
            ref = {'h': ((ch - ch.mean(1, keepdim=True)) ** 2).mean(2), 'l': ((cl - cl.mean(1, keepdim=True)) ** 2).mean(2)}

            def errors(a, b, fit=False):                                 # masks of the HSI and of the LiDAR input
                x = model.fill(h, l, a, b)
                zh, zl = adopt(lambda: model.encode(*x)) if fit else model.encode(*x)
                return {'h': ((model.f['h'](zh) - ch) ** 2).mean(2), 'l': ((model.f['l'](zl) - cl) ** 2).mean(2)}

            ea = errors(mh, ml, fit=True)                # A; its statistics stay for the passes below (training)
            eb = errors(torch.maximum(mh, sl), torch.maximum(ml, sh))    # the other modality hidden at sh, sl as well
            for m, own, sel in (('h', mh, sh), ('l', ml, sl)):
                for k, e, where in (('train', ea[m], own), ('visible', eb[m], own - sel), ('hidden', eb[m], sel)):
                    acc[m]['position'][k] += [float((e * where).sum()), float((ref[m] * where).sum()), float(where.sum())]

            none, every = torch.zeros_like(mh), torch.ones_like(mh)
            state = {'h': {'shown': errors(mh, none)['h'], 'train': ea['h'], 'hidden': errors(mh, every)['h']},
                     'l': {'shown': errors(none, ml)['l'], 'train': ea['l'], 'hidden': errors(every, ml)['l']}}
            for m, own in (('h', mh), ('l', ml)):
                k3, n = 3 * own.sum(1, keepdim=True), own.shape[1]       # the masked share against 1/3 and 2/3, exactly
                bins = {'all': torch.ones_like(k3), 'low': (k3 < n).to(own),
                        'mid': ((k3 >= n) & (k3 < 2 * n)).to(own), 'high': (k3 >= 2 * n).to(own)}
                for s in STATES:
                    for b in BINS:
                        w = own * bins[b]
                        acc[m]['modality'][s][b] += [float((state[m][s] * w).sum()), float(w.sum())]
    return acc


def summarise(acc):
    """The errors per masked position and the ratios of one model, per modality (nan where a group is empty)."""
    div = lambda a, b: float(a / b) if b > 0 else float('nan')                       # noqa: E731
    out = {}
    for m in 'hl':
        p, q = acc[m]['position'], acc[m]['modality']
        e1 = {k: div(p[k][0], p[k][2]) for k in GROUPS}
        e2 = {s: {b: div(q[s][b][0], q[s][b][1]) for b in BINS} for s in STATES}
        out[m] = dict(
            position=dict(error=e1, r2={k: 1 - div(p[k][0], p[k][1]) for k in GROUPS},
                          positions={k: int(p[k][2]) for k in GROUPS}, ratio=div(e1['hidden'], e1['visible'])),
            modality=dict(error=e2, positions={b: int(q['train'][b][1]) for b in BINS},
                          ratio={b: div(e2['hidden'][b], e2['train'][b]) for b in BINS},
                          ratio_shown={b: div(e2['hidden'][b], e2['shown'][b]) for b in BINS}))
    return out


def verdict(means, mins, yes, some, no):
    if not means:
        return 'no model was found'
    if min(means) >= STRONG and min(mins) > 1:
        return '%s (ratio >= %.1f on every dataset, > 1 for every model)' % (yes, STRONG)
    if min(means) >= WEAK:
        return '%s (ratio >= %.1f on every dataset, but not the line above)' % (some, WEAK)
    return '%s (ratio below %.1f on a dataset)' % (no, WEAK)


def keep_earlier(f):
    """An output file of an earlier run gets a name with its time instead of being overwritten."""
    if f.exists():
        f.replace(f.with_name('%s_%s%s' % (f.stem, time.strftime('%m%d_%H%M%S', time.localtime(f.stat().st_mtime)), f.suffix)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', default='full')
    ap.add_argument('--datasets', default='houston2013,muufl,trento')
    ap.add_argument('--seeds', help='e.g. 0,1 (default: the seeds of the plan)')
    ap.add_argument('--patches', type=int, default=1000, help='patches around test pixels per dataset')
    ap.add_argument('--chunk', type=int, default=250, help='patches per batch (the pool batch of training has 256)')
    ap.add_argument('--norm', default='both', choices=('both',) + tuple(NORMS))
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--data-root')
    ap.add_argument('--smoke', action='store_true', help='read results/formal_smoke (the try-out runs): checks the script only')
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors='replace')
    except Exception:                                    # noqa: BLE001
        pass
    warnings.filterwarnings('ignore')
    torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    plan, device, scenes = formal.Plan(smoke=a.smoke), torch.device(a.device), {}
    seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else plan.seeds
    datasets = [d.strip() for d in a.datasets.split(',') if d.strip()]
    norms = list(NORMS) if a.norm == 'both' else [a.norm]
    print('completion check: version %s, weights of the reported epoch, %d patches per dataset, device %s, %d threads'
          % (a.version, a.patches, device, torch.get_num_threads()))
    print('models read from %s (nothing is written there)' % plan.out_root)
    print('per model: ratio of test 1 (same position) and of test 2 (whole modality), HSI tokens / LiDAR tokens\n')
    table, t0 = {nm: {ds: {} for ds in datasets} for nm in norms}, time.time()
    for ds in datasets:
        cfg = plan.config(ds, a.version)
        st = Setting(cfg, ds, plan.root, find_data_root(cfg, plan.root, a.data_root), device, scenes)
        test = torch.as_tensor(np.asarray(st.split['test']), dtype=torch.long)
        pick = torch.randperm(len(test), generator=torch.Generator().manual_seed(0))[:a.patches]
        idx = test[pick].to(device)
        for k in seeds:
            f = plan.out_root / ds / a.version / 'ckpt' / ('run%d_best.pt' % k)
            if not f.exists():
                print('   %-12s run %d: %s is not there, skipped' % (ds, k, f))
                continue
            model = build_model(cfg, st.scene)
            if getattr(model, 'f', None) is None:
                raise SystemExit('version %s has no completion heads (it was trained without the M2C branch)' % a.version)
            model.load_state_dict(torch.load(f, map_location='cpu'))
            model.to(device).eval()
            if 'training' in norms and model.cfg.get('bn_masked') != 'batch':
                raise SystemExit('this version normalises its masked views with the running statistics in training '
                                 '(bn_masked: %s): run with --norm test' % model.cfg.get('bn_masked'))
            for nm in norms:
                t1 = time.time()
                row = table[nm][ds][k] = summarise(measure(model, st.sampler, idx, a.chunk, nm))
                print('   %-12s run %d, %-8s  same position %.2f / %.2f   whole modality %.2f / %.2f   (%.0f s; %.0f s so far)'
                      % (ds, k, nm, row['h']['position']['ratio'], row['l']['position']['ratio'],
                         row['h']['modality']['ratio']['all'], row['l']['modality']['ratio']['all'],
                         time.time() - t1, time.time() - t0))
    out = plan.root / 'results' / 'analysis' / ('completion_check%s.json' % ('_smoke' if a.smoke else ''))
    out.parent.mkdir(parents=True, exist_ok=True)
    keep_earlier(out)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(dict(version=a.version, patches=a.patches, seeds=seeds, strong=STRONG, weak=WEAK,
                       results={nm: {d: {str(k): v for k, v in r.items()} for d, r in t.items()} for nm, t in table.items()}),
                  f, indent=1)

    pm = lambda v: '%.2f +- %.2f' % (np.nanmean(v), np.nanstd(v))                    # noqa: E731
    print('\n' + '=' * 118)
    print('SUMMARY of the completion check: version %s%s' % (a.version, ' (TRY-OUT RUNS, the numbers mean nothing)' if a.smoke else ''))
    print('error = L_feat per masked position (x 1000), mean over the models; ratio: mean +- std over the models')
    verdicts = []
    for nm in norms:
        print('\n' + '-' * 118 + '\n' + NORMS[nm])
        print('\ntest 1, the same position: the other modality hidden at half of the masked positions; '
              'ratio = error where hidden / where visible')
        print('%-7s %-12s %4s  %-13s %-11s %-10s %-14s %-10s %s' % (
            'tokens', 'dataset', 'runs', 'A: as trained', 'B: visible', 'B: hidden', 'ratio', 'min ratio', 'R2 of A'))
        for m in 'hl':
            means, mins = [], []
            for ds in datasets:
                rows = [r[m]['position'] for r in table[nm][ds].values()]
                if not rows:
                    continue
                e = {g_: 1000 * np.mean([r['error'][g_] for r in rows]) for g_ in GROUPS}
                ratio = [r['ratio'] for r in rows]
                means.append(float(np.mean(ratio)))
                mins.append(float(np.min(ratio)))
                print('%-7s %-12s %4d  %-13.2f %-11.2f %-10.2f %-14s %-10.2f %.2f' % (
                    NAMES[m], ds, len(rows), e['train'], e['visible'], e['hidden'], pm(ratio), np.min(ratio),
                    np.mean([r['r2']['train'] for r in rows])))
            verdicts.append((nm, 'test 1, %s tokens' % NAMES[m], verdict(
                means, mins, 'RELIES ON the value of the other modality at the same position',
                'relies on the value at the same position TO A LIMITED DEGREE',
                'DOES NOT RELY on the value of the other modality at the same position')))
        print('\ntest 2, the whole modality: the other modality shown everywhere / as in training / hidden everywhere; '
              'ratio = error hidden / as in training')
        print('%-7s %-12s %4s  %-9s %-11s %-9s %-14s %-10s %s' % (
            'tokens', 'dataset', 'runs', 'shown', 'as trained', 'hidden', 'ratio', 'min ratio',
            'ratio where the own modality is masked < 1/3, 1/3 to 2/3, >= 2/3'))
        for m in 'hl':
            means, mins = [], []
            for ds in datasets:
                rows = [r[m]['modality'] for r in table[nm][ds].values()]
                if not rows:
                    continue
                e = {s: 1000 * np.mean([r['error'][s]['all'] for r in rows]) for s in STATES}
                ratio = [r['ratio']['all'] for r in rows]
                means.append(float(np.mean(ratio)))
                mins.append(float(np.min(ratio)))
                print('%-7s %-12s %4d  %-9.2f %-11.2f %-9.2f %-14s %-10.2f %s' % (
                    NAMES[m], ds, len(rows), e['shown'], e['train'], e['hidden'], pm(ratio), np.min(ratio),
                    ',  '.join('%.2f' % np.nanmean([r['ratio'][b] for r in rows]) for b in BINS[1:])))
            verdicts.append((nm, 'test 2, %s tokens' % NAMES[m], verdict(
                means, mins, 'USES the other modality', 'uses the other modality TO A LIMITED DEGREE',
                'DOES NOT USE the other modality')))
    print('\n' + '-' * 118)
    print('the rule (written before the first run of each test): yes = ratio >= %.1f on every dataset and > 1 for every '
          'model;\nto a limited degree = ratio >= %.1f on every dataset; no = otherwise' % (STRONG, WEAK))
    for nm in norms:
        print(NORMS[nm])
        for n_, name, text in verdicts:
            if n_ == nm:
                print('   %-22s %s' % (name + ':', text))
    print('written: %s   (%.0f s in all)' % (out, time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
