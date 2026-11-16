"""Train CNN_H on each labeled patch and one complementary masked copy."""
from functools import partial

import torch

from symbionet import external
from symbionet.corrupt import band_stats
from symbionet.model import bn_frozen, make_masks


class CNNHWithCopies(external.CoupledCNN):
    """Inference and the three classification heads are inherited from CNN_H."""

    def __init__(self, src, scene, patch, *, config, seed):
        super().__init__(src, scene, patch)
        model = config['model']
        if (not model['mask_labeled'] or model['decoder'] is not None or model['gate']
                or model['pre_gate'] or model['fill'] != 'mean' or model['bn_masked'] != 'batch'
                or model['mask_block'] != 1 or model['band_mask'] != 0
                or model['mask_mode'] != 'complementary'):
            raise ValueError('CNN_H transfer requires the copies-only pixel-masking configuration')
        self.mask_ratio = model['mask_ratio']
        self.mask_split = model['mask_split']
        self.masked_weight = config['training']['masked_weight']
        self.mask_generator = torch.Generator().manual_seed(seed + 1_000_003)

        # Apply the means in the image bands, before CNN_H projects HSI to its 20 PCs.
        for name, data in [('fill_h', scene.hsi), ('fill_l', scene.lidar)]:
            mean, _ = band_stats(data, scene.valid)
            value = torch.as_tensor(mean, dtype=torch.float32).reshape(1, -1, 1, 1)
            self.register_buffer(name, value, persistent=False)

    def masked_inputs(self, h, l):
        batch, _, height, width = h.shape
        mh, ml = make_masks(batch, height * width, self.mask_ratio, 'complementary',
                            self.mask_split, self.mask_generator, h.device)
        mh = mh.reshape(batch, 1, height, width)
        ml = ml.reshape(batch, 1, height, width)
        return h * (1 - mh) + self.fill_h * mh, l * (1 - ml) + self.fill_l * ml

    def training_loss(self, h, l, y, ce):
        clean = super().training_loss(h, l, y, ce)
        with bn_frozen(self):
            hm, lm = self.masked_inputs(h, l)
            masked = super().training_loss(hm, lm, y, ce)
        return clean + self.masked_weight * masked


def train(cfg, scene, sampler, split, seed, device, ext_root, log=print):
    """Run the original external training loop; both snapshots load into bare CNN_H."""
    name = 'cnnh_copies'
    if name in external.METHODS or name in external.SOURCES:
        raise RuntimeError(f'{name} is already registered in this process')
    factory = partial(CNNHWithCopies, config=cfg, seed=seed)
    external.METHODS[name] = factory
    external.SOURCES[name] = external.SOURCES['cnnh']
    try:
        return external.train(cfg, name, scene, sampler, split, seed, device, ext_root, log)
    finally:
        del external.SOURCES[name]
        del external.METHODS[name]
