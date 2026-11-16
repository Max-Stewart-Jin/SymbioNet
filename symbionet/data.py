"""Scenes, train/test splits and patch sampling.

A scene is one co-registered HSI + LiDAR image pair with its ground-truth map, read from the original .mat files
(layout of the `dataset` folder: Trento/, Houston13/, MUUFL/). Preprocessing, fixed per scene (the defaults reproduce
the patches of Geo-DeltaSSM exactly):
  * HSI and LiDAR: per-band min-max (or z-score) with statistics over the whole image; optional PCA of the HSI;
  * no-data pixels (all-zero spectra, or one spectrum repeated >= 50 times) are excluded from the unlabeled pool.
Splits are saved as .npz files of flat pixel indices, so every run and every method uses exactly the same training
pixels: the Geo-DeltaSSM splits (shipped in splits/), or new random ones (per class, n_train = ceil(ratio * n_class)).
Patches (p x p, centred on a pixel; zero or reflect padding at the image border) are gathered on the GPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# file, variable name, value of unlabeled pixels, class names (in label order 1..K)
DATASETS = {
    'trento': dict(
        hsi=('Trento/HSI_Trento.mat', 'HSI_Trento'),
        lidar=('Trento/Lidar_Trento.mat', 'Lidar_Trento'),
        gt=('Trento/GT_Trento.mat', 'GT_Trento'),
        unlabeled=0,
        classes=['Apple Trees', 'Buildings', 'Ground', 'Wood', 'Vineyard', 'Roads']),
    'houston2013': dict(
        hsi=('Houston13/Houston.mat', 'img'),
        lidar=('Houston13/LiDAR.mat', 'dsm'),
        gt=('Houston13/Houston_GT.mat', 'Houston_gt'),
        unlabeled=0,
        classes=['Healthy Grass', 'Stressed Grass', 'Synthetic Grass', 'Tree', 'Soil', 'Water', 'Residential',
                 'Commercial', 'Road', 'Highway', 'Railway', 'Parking Lot 1', 'Parking Lot 2', 'Tennis Court',
                 'Running Track']),
    'muufl': dict(
        hsi=('MUUFL/MUUFL_dataset.mat', 'HSI'),
        lidar=('MUUFL/MUUFL_dataset.mat', 'LiDAR'),
        gt=('MUUFL/MUUFL_dataset.mat', 'Label'),
        unlabeled=-1,
        classes=['Trees', 'Mostly Grass', 'Mixed Ground Surface', 'Dirt and Sand', 'Road', 'Water',
                 'Building Shadow', 'Building', 'Sidewalk', 'Yellow Curb', 'Cloth Panels']),
}
NODATA_REPEATS = 50          # one spectrum shared by >= this many pixels = a fill value


def load_mat_array(path: Path, key: str) -> np.ndarray:
    """One variable of a MATLAB file (v5 through scipy; v7.3 through h5py, transposed back to MATLAB order)."""
    from scipy.io import loadmat
    try:
        return np.asarray(loadmat(str(path), variable_names=[key])[key])
    except NotImplementedError:
        import h5py
        with h5py.File(path, 'r') as f:
            return np.asarray(f[key]).T


@dataclass
class Scene:
    name: str
    hsi: np.ndarray                 # (C, H, W) float32, preprocessed
    lidar: np.ndarray               # (Cl, H, W) float32, preprocessed
    labels: np.ndarray              # (H, W) int64, -1 = unlabeled, 0..K-1 = class
    valid: np.ndarray               # (H, W) bool, False = no-data pixel
    classes: list = field(default_factory=list)

    @property
    def shape(self):
        return self.labels.shape

    @property
    def n_classes(self):
        return len(self.classes)

    def summary(self) -> str:
        counts = np.bincount(self.labels[self.labels >= 0], minlength=self.n_classes)
        return ('%s: %d x %d pixels, HSI %d channels, LiDAR %d, %d classes, %d labeled, %d no-data'
                % (self.name, *self.shape, self.hsi.shape[0], self.lidar.shape[0], self.n_classes, counts.sum(),
                   (~self.valid).sum()))


def _channels_last(a: np.ndarray, shape_hw) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2:
        a = a[..., None]
    if a.shape[:2] != tuple(shape_hw):
        raise ValueError('image of shape %s does not match the ground truth %s' % (a.shape, shape_hw))
    return a


def nodata_mask(raw: np.ndarray) -> np.ndarray:
    """(H, W) True where the spectrum is all zeros or a fill value repeated >= NODATA_REPEATS times."""
    H, W, C = raw.shape
    rows = np.ascontiguousarray(raw.reshape(-1, C))
    zero = ~rows.any(1)
    keys = rows.view(np.dtype((np.void, rows.dtype.itemsize * C))).ravel()
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (zero | (counts[inverse.ravel()] >= NODATA_REPEATS)).reshape(H, W)


def normalise(x: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """Per-channel normalisation of x (H, W, C) with statistics over the pixels selected by mask (H, W)."""
    v = x[mask].astype(np.float64)
    if mode == 'zscore':
        shift, scale = v.mean(0), np.maximum(v.std(0), 1e-6)
    elif mode == 'minmax':
        shift, scale = v.min(0), np.maximum(v.max(0) - v.min(0), 1e-6)
    else:
        raise ValueError('normalisation must be "zscore" or "minmax"')
    return (x - shift.astype(np.float32)) / scale.astype(np.float32)          # float32 throughout


def pca(x: np.ndarray, valid: np.ndarray, k: int, scale: str = 'each', max_fit: int = 200_000,
        seed: int = 0) -> np.ndarray:
    """Project x (H, W, C) on its first k principal components (fitted on up to max_fit valid pixels). scale='each':
    every component standardised (like the bands without PCA); 'first': all divided by the standard deviation of the
    first component (keeps their relative scale)."""
    H, W, C = x.shape
    if not 0 < k <= C:
        raise ValueError('PCA needs 0 < k <= %d' % C)
    pts = x[valid]
    if len(pts) > max_fit:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_fit, replace=False)]
    pts = pts.astype(np.float64)
    mu = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(pts - mu, rowvar=False))
    comp = evecs[:, np.argsort(evals)[::-1][:k]]
    comp *= np.sign(comp[np.abs(comp).argmax(0), np.arange(k)])        # deterministic signs
    z = (x.reshape(-1, C) - mu.astype(np.float32)) @ comp.astype(np.float32)
    sd = z[valid.ravel()].std(0)
    if scale == 'each':
        z /= np.maximum(sd, 1e-6)
    elif scale == 'first':
        z /= max(float(sd[0]), 1e-6)
    else:
        raise ValueError('PCA scale must be "each" or "first"')
    return z.reshape(H, W, k)


def load_scene(name: str, root, hsi_norm: str = 'minmax', lidar_norm: str = 'minmax', pca_components: int = 0,
               pca_scale: str = 'each', seed: int = 0) -> Scene:
    """Read and preprocess one scene. `root` is the folder holding Trento/, Houston13/ and MUUFL/. Normalisation
    statistics are taken over the whole image (as Geo-DeltaSSM)."""
    if name not in DATASETS:
        raise ValueError('unknown dataset %r (choose from %s)' % (name, ', '.join(DATASETS)))
    spec, root = DATASETS[name], Path(root)
    gt = load_mat_array(root / spec['gt'][0], spec['gt'][1]).astype(np.int64)
    labels = np.where(gt == spec['unlabeled'], -1, gt - 1)
    if labels.min() < -1 or labels.max() >= len(spec['classes']):
        raise ValueError('unexpected label values in %s' % spec['gt'][0])
    raw = _channels_last(load_mat_array(root / spec['hsi'][0], spec['hsi'][1]), gt.shape)
    valid = ~nodata_mask(raw) | (labels >= 0)
    whole = np.ones(gt.shape, bool)
    hsi = normalise(raw, whole, hsi_norm)
    del raw
    if pca_components:
        hsi = pca(hsi, valid, pca_components, pca_scale, seed=seed)
    lidar = normalise(_channels_last(load_mat_array(root / spec['lidar'][0], spec['lidar'][1]), gt.shape), whole,
                      lidar_norm)
    return Scene(name, np.ascontiguousarray(hsi.transpose(2, 0, 1)), np.ascontiguousarray(lidar.transpose(2, 0, 1)),
                 labels, valid, list(spec['classes']))


# ------------------------------------------------------------------------------------------------------ splits
def n_train_per_class(counts, ratio) -> np.ndarray:
    """ceil(ratio * n) per class, in exact arithmetic (0.01 * 1300 must give 13, not 14)."""
    r = Fraction(str(ratio))
    return np.array([max(1, math.ceil(Fraction(int(n)) * r)) for n in counts], dtype=np.int64)


def make_split(labels: np.ndarray, ratio: float, seed: int) -> dict:
    """Per class, n_train = ceil(ratio * n_class) labeled pixels drawn at random; all other labeled pixels test."""
    rng = np.random.default_rng(seed)
    flat = labels.ravel()
    K = int(flat.max()) + 1
    counts = np.bincount(flat[flat >= 0], minlength=K)
    take = n_train_per_class(counts, ratio)
    train = np.concatenate([rng.choice(np.flatnonzero(flat == c), take[c], replace=False) for c in range(K)])
    train = np.sort(train)
    test = np.setdiff1d(np.flatnonzero(flat >= 0), train)
    return dict(train=train, test=test)


SPLITS = ('geo', 'random')


def split_path(folder, dataset: str, kind: str = 'geo', ratio: float = 0.01, seed: int = 0) -> Path:
    """'geo': the split of Geo-DeltaSSM (its training pixels, recovered from its patch files; shipped in splits/);
    'random': ceil(ratio * n_class) training pixels per class drawn with `seed`."""
    if kind == 'geo':
        return Path(folder) / ('%s_geodeltassm.npz' % dataset)
    if kind == 'random':
        return Path(folder) / ('%s_ratio%s_seed%d.npz' % (dataset, ('%g' % ratio).replace('.', 'p'), seed))
    raise ValueError('split must be one of %s' % (SPLITS,))


def get_split(scene: Scene, folder, kind: str = 'geo', ratio: float = 0.01, seed: int = 0, path=None) -> dict:
    """A saved split: `path`, or the default file of this kind in `folder`. A random split is created and saved on
    first use; a missing Geo-DeltaSSM split or a missing explicit file is an error."""
    explicit = path is not None
    path = Path(path) if explicit else split_path(folder, scene.name, kind, ratio, seed)
    if path.exists():
        z = np.load(path)
        split = dict(train=z['train'].astype(np.int64), test=z['test'].astype(np.int64))
        if tuple(z['shape']) != scene.shape:
            raise ValueError('split %s was made for an image of shape %s' % (path, tuple(z['shape'])))
    elif kind == 'random' and not explicit:
        split = make_split(scene.labels, ratio, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, train=split['train'], test=split['test'], shape=np.array(scene.shape), ratio=ratio, seed=seed)
    else:
        raise FileNotFoundError('split file not found: %s' % path)
    check_split(scene, split)
    split['path'] = str(path)
    return split


def check_split(scene: Scene, split: dict):
    flat = scene.labels.ravel()
    tr, te = split['train'], split['test']
    if np.intersect1d(tr, te).size or (flat[tr] < 0).any() or (flat[te] < 0).any():
        raise ValueError('a split must use disjoint, labeled pixels')
    if len(tr) + len(te) != (flat >= 0).sum():
        raise ValueError('train + test must cover every labeled pixel')


def pool_indices(scene: Scene, split: dict, with_test: bool = False, size: int = 0, seed: int = 0) -> np.ndarray:
    """Pixels the unlabeled pool may draw from (labels unused): every valid pixel except the test pixels.
    with_test: the test pixels too, i.e. every valid pixel (their spectra and heights only; the transductive setting).
    size > 0: a random subset of `size` of those pixels, drawn with `seed` (the same subset for every run); a size of
    at least the whole pool keeps the whole pool."""
    if size < 0:
        raise ValueError('pool_size must be 0 (the whole pool) or a positive number of pixels')
    mask = scene.valid.ravel().copy()
    if not with_test:
        mask[split['test']] = False
    pool = np.flatnonzero(mask)
    if 0 < size < len(pool):
        pool = np.sort(np.random.default_rng(seed).choice(pool, int(size), replace=False))
    return pool


def subset_train(scene: Scene, train: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """A part of the training pixels: per class ceil(fraction * n_class) of them (at least 1), drawn with `seed`.
    The test pixels are not touched; the training pixels left out stay in the unlabeled pool."""
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction must be in (0, 1]')
    if fraction >= 1:
        return train
    rng = np.random.default_rng(seed)
    y = scene.labels.ravel()[train]
    keep = []
    for c in np.unique(y):
        idx = train[y == c]
        keep.append(rng.choice(idx, int(n_train_per_class([len(idx)], fraction)[0]), replace=False))
    return np.sort(np.concatenate(keep))


def class_counts(scene: Scene, idx: np.ndarray) -> np.ndarray:
    return np.bincount(scene.labels.ravel()[idx], minlength=scene.n_classes)


# ----------------------------------------------------------------------------------------------------- patches
AUGMENTATIONS = ('none', 'flip', 'dihedral')


PADDINGS = ('zero', 'reflect')


class PatchSampler:
    """p x p patches around flat pixel indices, gathered from padded images kept on the device. padding='zero'
    (as Geo-DeltaSSM) or 'reflect' (mirror without repeating the border pixel) outside the image."""

    def __init__(self, scene: Scene, patch: int, device, padding: str = 'zero'):
        if patch % 2 == 0:
            raise ValueError('the patch size must be odd')
        if padding not in PADDINGS:
            raise ValueError('padding must be one of %s' % (PADDINGS,))
        r = patch // 2
        pad = (r, r, r, r)
        kw = dict(mode='constant', value=0.0) if padding == 'zero' else dict(mode='reflect')
        self.hsi = F.pad(torch.from_numpy(scene.hsi)[None], pad, **kw)[0].to(device)
        self.lidar = F.pad(torch.from_numpy(scene.lidar)[None], pad, **kw)[0].to(device)
        self.labels = torch.from_numpy(scene.labels.ravel()).to(device)
        self.W, self.patch, self.device = scene.shape[1], patch, device
        self.offsets = torch.arange(patch, device=device)

    def __call__(self, idx: torch.Tensor, augment: str = 'none', g: torch.Generator = None):
        """idx: (B,) flat pixel indices on the device -> HSI (B, C, p, p), LiDAR (B, Cl, p, p)."""
        rows = (idx // self.W)[:, None] + self.offsets
        cols = (idx % self.W)[:, None] + self.offsets
        ri, ci = rows[:, :, None], cols[:, None, :]
        h = self.hsi[:, ri, ci].permute(1, 0, 2, 3)
        l = self.lidar[:, ri, ci].permute(1, 0, 2, 3)
        if augment != 'none':
            h, l = self._augment(h, l, augment, g)
        return h.contiguous(), l.contiguous()

    def _augment(self, h, l, mode, g):
        """Random flips (mode 'flip') or all 8 symmetries of the square (mode 'dihedral'), the same for both
        modalities of a patch. The random numbers come from generator g (on the CPU), so runs are reproducible."""
        if mode not in AUGMENTATIONS:
            raise ValueError('augment must be one of %s' % (AUGMENTATIONS,))
        n_ops = 3 if mode == 'dihedral' else 2
        u = torch.rand(h.shape[0], n_ops, generator=g).to(h.device) < 0.5
        for k, op in enumerate((lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.transpose(-1, -2))[:n_ops]):
            m = u[:, k].view(-1, 1, 1, 1)
            h, l = torch.where(m, op(h), h), torch.where(m, op(l), l)
        return h, l

    def targets(self, idx: torch.Tensor) -> torch.Tensor:
        return self.labels[idx]
