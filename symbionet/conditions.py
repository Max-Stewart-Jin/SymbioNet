"""Test conditions beyond the corruptions of corrupt.py, and the named suites of conditions of the test stages.

corrupt.py defines the five corruptions behind the 46 conditions of the paper (noise, lost pixels, occluded blocks,
lost HSI bands, a missing modality). It is left as it is: training reads the band means through it, so its hash is
part of every training record. This module adds two kinds of condition on top of it:

    Shift    the LiDAR raster is moved by `level` pixels against the HSI and the labels (a registration error). The
             four directions (up, down, left, right) are the four draws of the condition, so its score is their
             mean. The border that the shift uncovers repeats the nearest row or column.
    Refill   a pixel or block condition of corrupt.py whose lost values are filled in another way than by the band
             means: 'nearest' = the value of the nearest pixel that is not lost (nearest-neighbour interpolation).
             The lost pixels are exactly those of the original condition and draw, so the two ways of filling are
             compared on the same losses.

Every condition has a label (the key of its scores), a test, a target and a level. apply() makes the test scene of
any condition; suite() names the lists that formal.py runs:

    corruptions   the 46 conditions of the paper
    shift         the LiDAR shifted by 1, 2 and 3 pixels
    interp        lost LiDAR pixels and blocks, filled by nearest-neighbour interpolation
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .corrupt import TESTS, Condition, conditions, corrupt_scene

PAPER_TESTS = ('noise', 'pixel', 'block', 'bands', 'missing')      # the tests behind the 46 conditions
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))                    # up, down, left, right as (rows, columns)
SHIFTS = (1, 2, 3)                                                 # pixels
FILLS = ('nearest',)
SUITES = ('corruptions', 'shift', 'interp')


@dataclass(frozen=True)
class Shift:
    level: float                                                   # pixels
    test: str = 'shift'
    target: str = 'lidar'

    @property
    def label(self) -> str:
        return 'shift lidar %g' % self.level


@dataclass(frozen=True)
class Refill:
    base: Condition                                                # a pixel or block condition of corrupt.py
    fill: str = 'nearest'

    def __post_init__(self):
        if self.base.test not in ('pixel', 'block'):
            raise ValueError('only lost pixels and blocks can be filled in another way, not %r' % self.base.test)
        if self.fill not in FILLS:
            raise ValueError('unknown fill %r (choose from %s)' % (self.fill, ', '.join(FILLS)))

    test = property(lambda self: self.base.test)
    target = property(lambda self: self.base.target)
    level = property(lambda self: self.base.level)

    @property
    def label(self) -> str:
        return '%s %s' % (self.base.label, self.fill)


def suite(name: str) -> list:
    """The conditions of a named suite, in the order of the summary."""
    if name == 'corruptions':
        return conditions(PAPER_TESTS)
    if name == 'shift':
        return [Shift(float(k)) for k in SHIFTS]
    if name == 'interp':
        return [Refill(c) for c in conditions(('pixel', 'block')) if c.target == 'lidar']
    raise ValueError('unknown suite %r (choose from %s)' % (name, ', '.join(SUITES)))


def n_draws(cond) -> int:
    """How many test scenes a condition has; its score is the mean over them."""
    return len(DIRECTIONS) if isinstance(cond, Shift) else TESTS[cond.test]['draws']


def shift_image(x: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """x (C, H, W) moved down by `rows` and right by `cols` pixels (negative: up, left); the border that the move
    uncovers repeats the nearest row or column."""
    _, H, W = x.shape
    r = np.clip(np.arange(H) - rows, 0, H - 1)
    c = np.clip(np.arange(W) - cols, 0, W - 1)
    return np.ascontiguousarray(x[:, r][:, :, c])


def lost_pixels(scene, cond: Condition, seed: int = 0, block: int = 16, draw: int = 0) -> dict:
    """{'hsi': mask, 'lidar': mask}: the pixels (H, W) that a pixel or block condition of corrupt.py loses in every
    modality. corrupt.py is asked to corrupt an image of ones with the fill value zero, so the masks are exactly the
    ones behind corrupt_scene(scene, cond, ...) with the same seed, block and draw."""
    if cond.test not in ('pixel', 'block'):
        raise ValueError('%r does not lose whole pixels' % cond.test)
    ones = np.ones((1,) + scene.shape, np.float32)
    zero = (np.zeros(1), np.ones(1))                                # (mean, variance) of the marker image
    marked = corrupt_scene(replace(scene, hsi=ones, lidar=ones), cond, dict(hsi=zero, lidar=zero), seed, block, draw)
    return dict(hsi=marked.hsi[0] == 0, lidar=marked.lidar[0] == 0)


def nearest_fill(x: np.ndarray, lost: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """A copy of x (C, H, W) in which every lost pixel (H, W) takes the values of the nearest pixel that is neither
    lost nor a no-data pixel (Euclidean distance on the pixel grid)."""
    from scipy import ndimage
    no_source = lost | ~valid
    if no_source.all():
        raise ValueError('every pixel is lost: nothing to interpolate from')
    near = ndimage.distance_transform_edt(no_source, return_distances=False, return_indices=True)
    y = x.copy()
    y[:, lost] = x[:, near[0][lost], near[1][lost]]
    return y


def apply(scene, cond, stats: dict, seed: int = 0, block: int = 16, draw: int = 0):
    """A copy of `scene` under any condition (draw number `draw`); stats as for corrupt.corrupt_scene. The input is
    not modified."""
    if isinstance(cond, Shift):
        rows, cols = DIRECTIONS[draw]
        k = int(cond.level)
        if k != cond.level or k < 0:
            raise ValueError('the shift is a whole number of pixels, got %r' % cond.level)
        return replace(scene, lidar=shift_image(scene.lidar, k * rows, k * cols))
    if isinstance(cond, Refill):
        lost = lost_pixels(scene, cond.base, seed, block, draw)
        out = {m: nearest_fill(getattr(scene, m), lost[m], scene.valid) if lost[m].any() else getattr(scene, m)
               for m in ('hsi', 'lidar')}
        return replace(scene, hsi=out['hsi'], lidar=out['lidar'])
    return corrupt_scene(scene, cond, stats, seed, block, draw)
