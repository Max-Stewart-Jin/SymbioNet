"""Published HSI-LiDAR networks inside the pipeline of this project.

The comparison tables of the paper give the accuracy of the compared methods on clean data. To see how published
networks behave under the test conditions of this project (noise, lost pixels, occluded blocks, lost bands, a
missing modality), a few of them whose authors publish code are trained and tested here (compare.py). They get the
same split and the same input patches as SymbioNet (the same normalisation; in a test the same corrupted scene) and
are trained with the protocol of the base network: optimizer, learning-rate schedule, epochs, batch size, patch
size, augmentation, label smoothing and class weights. Each keeps what belongs to the method: its network, its own
preprocessing of a patch, its loss terms and its rule for the final prediction.

The code of these networks is not part of this repository and is not copied into it. It is read from the files
that the authors publish. Download the repositories and put them under external/ (any folder names):

    dahgm    Xie et al., TGRS 2025     github.com/RogsXie/DAHGMN                             model/DAHGM.py
    cnnh     Hang et al., TGRS 2020    github.com/RenlongHang/Coupled-CNNs                   CoupledCNNs.py
    endnet   Hong et al., GRSL 2022    github.com/likyoo/Multimodal-Remote-Sensing-Toolkit   model/EndNet.py
    mft      Roy et al., TGRS 2023     github.com/AnkurDeria/MFT                             Model/MFT.ipynb

Only the definitions of the network classes are used. CoupledCNNs.py and the notebook of MFT are scripts that also
load data and train when they are run, so their classes are cut out of the source text and nothing else of these
files is executed (definitions()). DAHGM.py and EndNet.py are modules and are imported as they are. DAHGM.py needs
the package mamba_ssm, which is built for Linux; where it cannot be imported, symbionet/mamba_compat.py takes its
place (the same block in plain PyTorch), and the record of every run says which of the two was used.

SOURCES holds the SHA-256 of the files this module was written and tested with. Another version of a file is used
all the same, with a warning in the log and in the record.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import mamba_compat
from .data import class_counts
from .engine import _batches, class_weights, evaluate, make_optimizer, predict, set_seed

SOURCES = {
    'dahgm': dict(name='DAHGM', file='model/DAHGM.py', repo='https://github.com/RogsXie/DAHGMN',
                  sha256='c8a039ed96814dd3d6d91f55ef1ab68455066ba01458471dda2b738b30a10566'),      # 2026-07-10
    'cnnh': dict(name='CNN_H (Coupled CNNs)', file='CoupledCNNs.py', repo='https://github.com/RenlongHang/Coupled-CNNs',
                 sha256='d3ba07830941037f12ca074d6ab954b6830ae2e5022674e5c34f12a9f09c96cf'),       # 2020-06-16
    'endnet': dict(name='EndNet', file='model/EndNet.py',
                   repo='https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit',
                   sha256='86eb2075da915a7af3c82dcb6f3334ae98e18d6ffdf7b8e7d53d6be17c815b48'),     # 2022-01-23
    'mft': dict(name='MFT', file='Model/MFT.ipynb', repo='https://github.com/AnkurDeria/MFT',
                sha256='3c58c429ba471ed6e69883bbca1171d1b7ddd70864fda9f2f6e0ec70e50615ea'),        # 2023-12-23
}


# ---------------------------------------------------------------------------------------------- reading the sources
def source_hash(path) -> str:
    """SHA-256 of a text file with the line endings ignored, so Windows and Linux copies agree."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def locate(method: str, ext_root) -> tuple:
    """(path of the source file of a method under ext_root, warning or ''). A copy with the expected hash is
    preferred when the file is found more than once."""
    src = SOURCES[method]
    ext_root = Path(ext_root)
    found = sorted(p for p in ext_root.rglob(Path(src['file']).name)
                   if p.is_file() and p.as_posix().endswith('/' + src['file'])) if ext_root.is_dir() else []
    if not found:
        raise FileNotFoundError('%s: %s was not found under %s. Download %s and put the folder there.'
                                % (src['name'], src['file'], ext_root, src['repo']))
    for p in found:
        if source_hash(p) == src['sha256']:
            return p, ''
    return found[0], ('%s is not the version this project was tested with (SHA-256 %s..., tested: %s...)'
                      % (found[0], source_hash(found[0])[:10], src['sha256'][:10]))


def definitions(source: str, space: dict, filename: str = '<published source>') -> dict:
    """Run only the class and function definitions at the top level of `source`, in the namespace `space`. The
    rest of the source (loading data, training, printing) is not executed."""
    tree = ast.parse(source, filename)
    keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(keep, []), filename, 'exec'), space)
    return space


def notebook_cell(path, marker: str) -> str:
    """The source of the first code cell of a Jupyter notebook that contains `marker` (shell and magic lines
    dropped)."""
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    for cell in nb['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and marker in text:
            return '\n'.join(line for line in text.split('\n') if not line.lstrip().startswith(('!', '%')))
    raise ValueError('%s has no code cell with %r' % (path, marker))


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:                    # noqa: BLE001 - also a package that is installed but cannot be loaded
        return False


@contextmanager
def stand_ins():
    """While a published module is imported: mamba_ssm and thop are provided if the real packages are missing
    (mamba_ssm by symbionet.mamba_compat; thop, which the modules use only to count operations, by an empty
    module). Yields {'mamba': which implementation the module gets}."""
    added, used = [], {}
    if _importable('mamba_ssm'):
        used['mamba'] = 'mamba_ssm %s' % getattr(sys.modules['mamba_ssm'], '__version__', '')
    else:
        sys.modules['mamba_ssm'] = mamba_compat
        added.append('mamba_ssm')
        used['mamba'] = 'symbionet.mamba_compat (plain PyTorch)'
    if not _importable('thop'):
        stub = types.ModuleType('thop')

        def profile(*args, **kwargs):
            raise RuntimeError('thop is not installed (pip install thop)')
        stub.profile = profile
        sys.modules['thop'] = stub
        added.append('thop')
    try:
        yield used
    finally:
        for name in added:
            sys.modules.pop(name, None)


def load_module(path, name: str) -> tuple:
    """(the published module at `path`, imported under `name`; {'mamba': ...})."""
    with stand_ins() as used:
        spec = importlib.util.spec_from_file_location(name, str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, used


def rearrange(x, pattern: str):
    """einops.rearrange where einops is installed; otherwise the one pattern the notebook of MFT uses."""
    try:
        from einops import rearrange as einops_rearrange
        return einops_rearrange(x, pattern)
    except ImportError:
        if pattern.replace(' ', '') == 'bhw->bwh':
            return x.transpose(1, 2)
        raise NotImplementedError('pattern %r needs einops (pip install einops)' % pattern)


def pca_basis(hsi: np.ndarray, valid: np.ndarray, k: int) -> tuple:
    """(mean (C,), components (k, C)) of the spectra of the valid pixels: the projection (x - mean) @ components.T
    is the PCA of sklearn without whitening. The sign of a component is fixed by its largest entry."""
    x = hsi[:, valid].astype(np.float64)
    mean = x.mean(1)
    x = x - mean[:, None]
    values, vectors = np.linalg.eigh(x @ x.T / max(x.shape[1] - 1, 1))
    comps = vectors[:, np.argsort(values)[::-1][:k]].T
    comps = comps * np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(1)])[:, None]
    return mean, comps


# ------------------------------------------------------------------------------------------------------- networks
class Compared(nn.Module):
    """A published network with the interface the training and test code of this project use: forward(h, l) gives
    the class scores of a batch of patches h (B, C, w, w), l (B, Cl, w, w)."""

    def __init__(self):
        super().__init__()
        self.cfg = {'mask_labeled': False}       # read by code written for SymbioNet: no masked view, no branch
        self.branch = False
        self.info = {}

    def training_loss(self, h, l, y, ce):
        return ce(self(h, l), y)

    def before_eval(self, batches):
        """Called before the model is scored; batches() yields (h, l, y) of the training patches."""

    def count_parameters(self) -> dict:
        return dict(test_time=sum(p.numel() for p in self.parameters() if p.requires_grad), training_only=0)

    def strip_training_branch(self):
        return self


class DAHGM(Compared):
    """DAHGM as published: the network takes the two patches and returns the weighted sum of three softmax
    outputs, which the published script gives to the cross-entropy loss; the same is done here."""

    def __init__(self, src, scene, patch):
        super().__init__()
        module, used = load_module(src, '_published_dahgm')
        self.info.update(used)
        self.net = module.DAHGM(scene.hsi.shape[0], scene.lidar.shape[0], scene.n_classes)

    def forward(self, h, l, on=None):
        return self.net(h, l)


class CoupledCNN(Compared):
    """Coupled CNNs as published: the HSI patch is reduced to 20 principal components (fitted on the image, as in
    the script), the two branches share two layers, and three outputs are trained with the weights 0.01, 0.01, 1.
    The prediction is the sum of the three outputs with class-wise weights, the share of each output in the
    correctly classified training pixels of the class; the weights are set anew before every scoring (before_eval)
    and are kept with the weights of the network."""
    PCS, FM, AUX = 20, 32, 0.01

    def __init__(self, src, scene, patch):
        super().__init__()
        k = scene.n_classes
        space = dict(nn=nn, torch=torch, NC=self.PCS, FM=self.FM, Classes=k)
        definitions(Path(src).read_text(encoding='utf-8'), space, str(src))
        self.net = space['CNN']()
        mean, comps = pca_basis(scene.hsi, scene.valid, self.PCS)
        self.register_buffer('pca_w', torch.tensor(comps, dtype=torch.float32)[:, :, None, None])
        self.register_buffer('pca_b', torch.tensor(-(comps @ mean), dtype=torch.float32))
        self.register_buffer('w', torch.cat([torch.zeros(2, k), torch.ones(1, k)]))

    def heads(self, h, l):
        return self.net(F.conv2d(h, self.pca_w, self.pca_b), l)

    def forward(self, h, l, on=None):
        o1, o2, o3 = self.heads(h, l)
        return self.w[2] * o3 + self.w[1] * o2 + self.w[0] * o1

    def training_loss(self, h, l, y, ce):
        o1, o2, o3 = self.heads(h, l)
        return self.AUX * ce(o1, y) + self.AUX * ce(o2, y) + ce(o3, y)

    @torch.no_grad()
    def before_eval(self, batches):
        was_training = self.training
        self.eval()
        right = torch.zeros_like(self.w)
        for h, l, y in batches():
            for i, o in enumerate(self.heads(h, l)):
                right[i] += torch.bincount(y[o.argmax(1) == y], minlength=right.shape[1]).to(right)
        self.train(was_training)
        self.w.copy_(right / (right.sum(0, keepdim=True) + 1e-5))


class EndNet(Compared):
    """EndNet (the PyTorch version of the toolkit): a pixel-wise encoder-decoder; its input is the centre pixel of
    the patch, and its loss is the cross-entropy plus the reconstruction errors of both inputs."""

    def __init__(self, src, scene, patch):
        super().__init__()
        module, _ = load_module(src, '_published_endnet')
        self.net = module.EndNet(scene.hsi.shape[0], scene.lidar.shape[0], scene.n_classes)
        self.centre = patch // 2

    def pixel(self, h, l):
        return h[:, :, self.centre, self.centre], l[:, :, self.centre, self.centre]

    def forward(self, h, l, on=None):
        return self.net(*self.pixel(h, l))[0]

    def training_loss(self, h, l, y, ce):
        out, d1, d2, x1, x2 = self.net(*self.pixel(h, l))
        return ce(out, y) + F.mse_loss(d1, x1) + F.mse_loss(d2, x2)

    def count_parameters(self) -> dict:
        decoder = sum(p.numel() for n, p in self.net.named_parameters() if n.startswith('decoder'))
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(test_time=total - decoder, training_only=decoder)


class MFT(Compared):
    """MFT with channel tokenization (the first model of the published notebook), FM = 16 as there."""
    FM = 16

    def __init__(self, src, scene, patch):
        super().__init__()
        space = dict(torch=torch, nn=nn, F=F, LayerNorm=nn.LayerNorm, Linear=nn.Linear, Dropout=nn.Dropout,
                     Softmax=nn.Softmax, copy=copy, rearrange=rearrange, patchsize=patch)
        definitions(notebook_cell(src, 'MFT WITH CHANNEL TOKENIZATION'), space, str(src))
        self.net = space['MFT'](self.FM, scene.hsi.shape[0], scene.lidar.shape[0], scene.n_classes, False)

    def forward(self, h, l, on=None):
        return self.net(h.flatten(2), l.flatten(2))


METHODS = {'dahgm': DAHGM, 'cnnh': CoupledCNN, 'endnet': EndNet, 'mft': MFT}


def build(method: str, scene, patch: int, ext_root) -> Compared:
    if method not in METHODS:
        raise ValueError('unknown method %r (choose from %s)' % (method, ', '.join(METHODS)))
    src, warning = locate(method, ext_root)
    model = METHODS[method](src, scene, patch)
    model.info.update(method=method, name=SOURCES[method]['name'], source=str(src), sha256=source_hash(src),
                      as_tested=not warning)
    if warning:
        model.info['warning'] = warning
    return model


# ------------------------------------------------------------------------------------------------------- training
def train(cfg: dict, method: str, scene, sampler, split: dict, seed: int, device, ext_root, log=print) -> dict:
    """Train one published network with the protocol of symbionet/engine.py (the settings in cfg are those of the
    base network) and return what engine.train_run returns: the scores at the best and the last epoch, the curves
    and the weights of both epochs. The loss of a step is the network's own (Compared.training_loss)."""
    set_seed(seed)
    t, e = cfg['training'], cfg['eval']
    model = build(method, scene, cfg['data']['patch'], ext_root).to(device)
    opt = make_optimizer(cfg, model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t['epochs'], eta_min=t['min_lr'])
    w = class_weights(class_counts(scene, split['train']), t['class_weight'])
    ce = nn.CrossEntropyLoss(weight=None if w is None else w.to(device), label_smoothing=t['label_smoothing'])

    train_idx = torch.from_numpy(split['train']).to(device)
    test_idx = torch.from_numpy(split['test']).to(device)
    if e.get('subset'):                                  # quick checks only: the part of the test set engine uses
        keep = np.random.default_rng(0).choice(len(test_idx), min(e['subset'], len(test_idx)), replace=False)
        test_idx = test_idx[torch.from_numpy(np.sort(keep)).to(device)]
    test_y = sampler.targets(test_idx)
    g_data = torch.Generator().manual_seed(seed)                 # batch order and augmentation

    def train_set():
        for part in train_idx.split(e['chunk']):
            yield (*sampler(part), sampler.targets(part))

    best = dict(oa=-1.0, epoch=0, state=None)
    history, start = [], time.time()
    for epoch in range(1, t['epochs'] + 1):
        model.train()
        total, steps = 0.0, 0
        for b in _batches(train_idx, t['batch_size'], g_data):
            h, l = sampler(b, cfg['data']['augment'], g_data)
            loss = model.training_loss(h, l, sampler.targets(b), ce)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total, steps = total + loss.detach(), steps + 1
        sched.step()
        if epoch % e['every'] == 0 or epoch == t['epochs']:
            model.before_eval(train_set)
            oa = 100 * (predict(model, sampler, test_idx, e['chunk']) == test_y).float().mean().item()
            history.append(dict(epoch=epoch, test_oa=oa, loss=(total / steps).item()))
            if oa > best['oa']:
                best = dict(oa=oa, epoch=epoch,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if epoch % cfg['print_every'] == 0 or epoch == t['epochs']:
                log('    epoch %4d  loss %.4f  test OA %6.2f  (best %6.2f at epoch %d)'
                    % (epoch, history[-1]['loss'], oa, best['oa'], best['epoch']))
    seconds = time.time() - start
    last = evaluate(model, sampler, test_idx, scene.n_classes, e['chunk'])
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best['state'])
    best_scores = evaluate(model, sampler, test_idx, scene.n_classes, e['chunk'])
    return dict(seed=seed, best_epoch=best['epoch'], best=best_scores, last=last, history=history, seconds=seconds,
                parameters=model.count_parameters(), compared=dict(model.info), state=best['state'],
                last_state=last_state)
