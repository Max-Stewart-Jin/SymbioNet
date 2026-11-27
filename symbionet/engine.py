"""Single-run training and classification evaluation for SymbioNet."""
import random
import time

import numpy as np
import torch
import torch.nn as nn

from .corrupt import band_stats
from .data import PatchSampler, class_counts
from .metrics import confusion, scores
from .model import SymbioNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(counts: np.ndarray, mode) -> torch.Tensor:
    """None, or per-class weights (mean 1): 'inv_sqrt' = 1 / sqrt(n_c), 'inv' = 1 / n_c."""
    if mode in (None, 'none'):
        return None
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
    if mode == 'inv_sqrt':
        w = 1 / np.sqrt(counts)
    elif mode == 'inv':
        w = 1 / counts
    else:
        raise ValueError('class_weight must be none, inv_sqrt or inv')
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, sampler: PatchSampler, idx: torch.Tensor, chunk: int) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = [model(*sampler(part)).argmax(1) for part in idx.split(chunk)]
    model.train(was_training)
    return torch.cat(out)


def evaluate(model, sampler, idx, n_classes, chunk) -> dict:
    cm = confusion(predict(model, sampler, idx, chunk), sampler.targets(idx), n_classes)
    return dict(scores(cm), confusion=cm.tolist())


def build_model(cfg: dict, scene) -> SymbioNet:
    model = SymbioNet(scene.hsi.shape[0], scene.lidar.shape[0], scene.n_classes, window=cfg['data']['patch'],
                      **cfg['model'])
    if model.cfg['fill'] == 'mean':                      # the band means that replace lost values (corrupt.py)
        model.set_fill(band_stats(scene.hsi, scene.valid)[0], band_stats(scene.lidar, scene.valid)[0])
    return model


def make_optimizer(cfg: dict, params):
    t = cfg['training']
    cls = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[t['optimizer']]
    return cls(params, lr=t['lr'], weight_decay=t['weight_decay'])


def _batches(idx: torch.Tensor, size: int, g: torch.Generator):
    perm = idx[torch.randperm(len(idx), generator=g).to(idx.device)]
    parts = list(perm.split(size))
    if len(parts) > 1 and len(parts[-1]) == 1:          # BatchNorm needs >= 2 samples in a batch
        parts[-2:] = [torch.cat(parts[-2:])]
    return parts


def train_run(cfg: dict, scene, sampler: PatchSampler, split: dict, pool: np.ndarray, seed: int, device,
              log=print) -> dict:
    """Train one model; return its scores at the best and the last epoch, the curves, and the weights of both
    epochs (`state`: the epoch with the best test OA; `last_state`: the last epoch)."""
    set_seed(seed)
    t, e = cfg['training'], cfg['eval']
    model = build_model(cfg, scene).to(device)
    opt = make_optimizer(cfg, model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['epochs'], eta_min=t['min_lr'])
    w = class_weights(class_counts(scene, split['train']), t['class_weight'])
    ce = nn.CrossEntropyLoss(weight=None if w is None else w.to(device), label_smoothing=t['label_smoothing'])
    mc = model.cfg

    train_idx = torch.from_numpy(split['train']).to(device)
    test_idx = torch.from_numpy(split['test']).to(device)
    if e.get('subset'):                                  # quick checks only: a fixed random part of the test set
        keep = np.random.default_rng(0).choice(len(test_idx), min(e['subset'], len(test_idx)), replace=False)
        test_idx = test_idx[torch.from_numpy(np.sort(keep)).to(device)]
    test_y = sampler.targets(test_idx)
    pool_idx = torch.from_numpy(pool).to(device) if model.branch else None
    g_data = torch.Generator().manual_seed(seed)                 # batch order, augmentation, pool draws
    g_mask = torch.Generator().manual_seed(seed + 1_000_003)     # masks of the training branch
    g_gate = torch.Generator().manual_seed(seed + 2_000_003)     # masked views the pre-gate acts on
    g_spec = torch.Generator().manual_seed(seed + 3_000_003)     # independent HSI spectral masking

    best = dict(oa=-1.0, epoch=0, state=None)
    history, start = [], time.time()
    for epoch in range(1, t['epochs'] + 1):
        model.train()
        sums, steps = {}, 0
        for b in _batches(train_idx, t['batch_size'], g_data):
            h, l = sampler(b, cfg['data']['augment'], g_data)
            y = sampler.targets(b)
            loss = ce(model(h, l), y)
            parts = {'ce': loss.detach()}
            if mc['mask_labeled']:                       # the same labels on a masked copy (BatchNorm statistics kept)
                with model.masked_bn():
                    ce_m = ce(model(*model.masked_inputs(h, l, g_mask),
                                    on=model.draw_gate(h.shape[0], g_gate, device)), y)
                mw = t.get('masked_weight', 1.0)                 # weight of the masked copy in the loss
                loss = loss + (ce_m if mw == 1 else mw * ce_m)
                parts['ce_m'] = ce_m.detach()
            if model.branch:
                pidx = pool_idx[torch.randint(len(pool_idx), (t['pool_batch'],), generator=g_data).to(device)]
                o = model.pool_losses(*sampler(pidx, cfg['data']['augment'], g_data), g=g_mask,
                                      on=model.draw_gate(t['pool_batch'], g_gate, device), spec_g=g_spec)
                loss = loss + mc['lam'] * o['rec']
                loss = loss + mc['beta'] * o['feat']
                parts.update({k: o[k].detach() for k in ('rec', 'mse_h', 'mse_l', 'feat')})
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
        sched.step()
        if epoch % e['every'] == 0 or epoch == t['epochs']:
            oa = 100 * (predict(model, sampler, test_idx, e['chunk']) == test_y).float().mean().item()
            row = dict(epoch=epoch, test_oa=oa, **{k: (v / steps).item() for k, v in sums.items()})
            history.append(row)
            if oa > best['oa']:
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best = dict(oa=oa, epoch=epoch, state=state)
            if epoch % cfg['print_every'] == 0 or epoch == t['epochs']:
                extra = ''.join('  %s %.4f' % (k, row[k]) for k in ('ce_m', 'rec', 'feat')
                                if k in row)
                log('    epoch %4d  ce %.4f%s  test OA %6.2f  (best %6.2f at epoch %d)'
                    % (epoch, row['ce'], extra, oa, best['oa'], best['epoch']))
    seconds = time.time() - start
    last = evaluate(model, sampler, test_idx, scene.n_classes, e['chunk'])
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best['state'])
    best_scores = evaluate(model, sampler, test_idx, scene.n_classes, e['chunk'])
    return dict(seed=seed, best_epoch=best['epoch'], best=best_scores, last=last, history=history,
                seconds=seconds, parameters=model.count_parameters(), state=best['state'], last_state=last_state)
