# Reproduction split indices — provenance

This directory includes the fixed Geo-ΔSSM split indices for Houston2013, MUUFL and Trento, used to reproduce
SymbioNet's manuscript protocol. Their **reuse for this public repository was confirmed by a collaborating
author**, as communicated by the SymbioNet project author on 2026-11-16.

- `houston2013_geodeltassm.npz`: 158 training, 14,871 test indices
- `muufl_geodeltassm.npz`: 541 training, 53,146 test indices
- `trento_geodeltassm.npz`: 306 training, 29,908 test indices

These files are **reconstructed index arrays**, obtained by matching the centres of Geo-ΔSSM's HSI–LiDAR
patch files, rather than copies of the original source scenes. Each NPZ contains `train`, `test`,
`shape`, `ratio`, `seed`, and `source`. The recorded seed is `-1` because the original sampling seed is
not recoverable from the patch arrays. The files have not been re-sampled for this release.

For attribution, cite the Geo-ΔSSM research by its authors in addition to SymbioNet when reusing this split.
These indices are provided for research reproducibility with the above consent; the repository's MIT
software license should not be read as a claim of ownership or relicensing of third-party materials.

The three `*_ratio0p01_seed0.npz` files in this directory are **different random splits**, included only
for testing or alternative training, and must not be described as the fixed manuscript splits.

No original hyperspectral, LiDAR, or ground-truth scene arrays are included in this directory.
