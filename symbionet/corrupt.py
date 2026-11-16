"""Test-time corruptions of a scene, for robustness experiments.

A corruption is applied to the preprocessed image before the patches are cut, so a corrupted pixel affects every patch
that contains it, as a real sensor defect or cloud would. One condition = (test, target, level):

    noise    additive white Gaussian noise; level = SNR in dB per band: sigma_b^2 = var_b / 10^(SNR / 10)
    pixel    a random fraction `level` of the pixels loses all its bands
    block    random square blocks of side `block` pixels cover a fraction `level` of the image (e.g. cloud cover)
    bands    a random fraction `level` of the HSI bands is lost everywhere (dead or noisy bands)
    missing  the whole modality is lost (level 1)

target: 'hsi', 'lidar' or 'both'. Lost values are replaced by the band means of the clean image (a model-agnostic
imputation, so every method sees the same input); band means and variances are taken over the valid pixels of the
clean image.

Randomness. Each (test, modality, draw) has its own random stream, shared by all the levels of the test:
  - the levels are nested: the pixels, blocks or bands lost at a level are also lost at every higher level, and the
    noise has the same pattern at every SNR, only scaled, so a curve over the levels changes only through the level;
  - 'both' corrupts each modality exactly as 'hsi' and 'lidar' do, independently of each other;
  - a test can be repeated with independent draws (TESTS[...]['draws']), and the scores averaged;
  - all models are tested on exactly the same corrupted images, and repeated runs give the same images.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

TARGETS = ('hsi', 'lidar', 'both')
MODALITIES = ('hsi', 'lidar')

# default levels, targets and number of random draws of every test (the order is also the order of the summary)
TESTS = {
    'noise': dict(levels=(30, 20, 10, 5, 0), targets=('hsi', 'lidar', 'both'), draws=1, unit='SNR (dB)'),
    'pixel': dict(levels=(0.1, 0.3, 0.5, 0.7, 0.9), targets=('hsi', 'lidar', 'both'), draws=1,
                  unit='fraction of pixels lost'),
    'block': dict(levels=(0.1, 0.2, 0.3, 0.4, 0.5), targets=('hsi', 'lidar'), draws=3,
                  unit='fraction of the image occluded'),
    'bands': dict(levels=(0.1, 0.3, 0.5, 0.7), targets=('hsi',), draws=5, unit='fraction of HSI bands lost'),
    'missing': dict(levels=(1,), targets=('hsi', 'lidar'), draws=1, unit='modality lost'),
}
RANDOM = ('noise', 'pixel', 'block', 'bands')          # tests whose result depends on the draw


@dataclass(frozen=True)
class Condition:
    test: str
    target: str
    level: float

    @property
    def label(self) -> str:
        return '%s %s %g' % (self.test, self.target, self.level)


def conditions(tests=None) -> list:
    """All conditions of the given tests (default: all), in summary order."""
    tests = list(TESTS) if tests is None else list(tests)
    unknown = set(tests) - set(TESTS)
    if unknown:
        raise ValueError('unknown tests %s (choose from %s)' % (sorted(unknown), ', '.join(TESTS)))
    return [Condition(t, m, float(v)) for t in tests for m in TESTS[t]['targets'] for v in TESTS[t]['levels']]


def band_stats(x: np.ndarray, valid: np.ndarray):
    """Per-band mean and variance of x (C, H, W) over the valid pixels (H, W), in float64."""
    v = x[:, valid].astype(np.float64)
    return v.mean(1), v.var(1)


def _rng(test: str, modality: str, draw: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, list(TESTS).index(test), MODALITIES.index(modality), draw])


def _corrupt_one(x: np.ndarray, test: str, level: float, mean, var, rng, block: int) -> np.ndarray:
    """One modality x (C, H, W) -> a corrupted copy. The random draws do not depend on the level (nested levels)."""
    C, H, W = x.shape
    y = x.copy()
    fill = mean.astype(np.float32)[:, None]
    if test == 'noise':
        sigma = np.sqrt(var / 10 ** (level / 10)).astype(np.float32)
        y += rng.standard_normal(x.shape, dtype=np.float32) * sigma[:, None, None]
    elif test == 'pixel':
        lost = rng.random((H, W)) < level             # one uniform value per pixel: nested in the level
        y[:, lost] = fill
    elif test == 'block':
        if not 0 <= level < 1:
            raise ValueError('block coverage must be in [0, 1); use the missing test for a lost modality')
        lost = np.zeros((H, W), bool)
        s, covered, need = int(min(block, H, W)), 0, level * H * W
        while covered < need:                         # the same sequence of blocks at every level, cut earlier
            r, c = rng.integers(0, H - s + 1), rng.integers(0, W - s + 1)
            covered += int((~lost[r:r + s, c:c + s]).sum())
            lost[r:r + s, c:c + s] = True
        y[:, lost] = fill
    elif test == 'bands':
        lost = rng.permutation(C)[:int(round(level * C))]     # a prefix of one random order: nested in the level
        y[lost] = mean[lost].astype(np.float32)[:, None, None]
    elif test == 'missing':
        y[:] = mean.astype(np.float32)[:, None, None]
    else:
        raise ValueError('unknown test %r' % test)
    return y


def corrupt_scene(scene, cond: Condition, stats: dict, seed: int = 0, block: int = 16, draw: int = 0):
    """A copy of `scene` (symbionet.data.Scene) with the condition applied (random draw number `draw`);
    stats = {'hsi': (mean, var), 'lidar': (mean, var)} from band_stats of the clean scene. The input is not modified."""
    if cond.test == 'bands' and cond.target != 'hsi':
        raise ValueError('the bands test applies to the HSI only')
    out = {}
    for m in MODALITIES:
        x = getattr(scene, m)
        hit = cond.target in (m, 'both')
        out[m] = _corrupt_one(x, cond.test, cond.level, *stats[m], _rng(cond.test, m, draw, seed), block) if hit else x
    return replace(scene, hsi=out['hsi'], lidar=out['lidar'])
