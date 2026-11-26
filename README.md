# SymbioNet

**Symbiotic Cross-Modal Completion Network for Hyperspectral and LiDAR Joint Classification**

SymbioNet combines a gated symmetric fusion encoder, masked labeled counterparts, and masked mutual completion (M2C) on unlabeled patches for HSI–LiDAR classification. The completion branch is used only during training.

## Installation

Python 3.9–3.12 and PyTorch 2.1 or later are required. A CUDA-enabled PyTorch build is recommended for training.

```bash
pip install -r requirements.txt
python formal.py plan
```

See [Quick start](docs/QUICKSTART.md) for environment setup and a smoke test.

## Data

Obtain the datasets from their providers under the applicable terms:

- Houston2013: [IEEE GRSS Data Fusion Contest](https://www.grss-ieee.org/community/technical-committees/2013-ieee-grss-data-fusion-contest/) and [University of Houston](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/).
- MUUFL: [GatorSense MUUFL Gulfport](https://github.com/GatorSense/MUUFLGulfport).
- Trento: [University of Trento Remote Sensing Laboratory](https://rslab.disi.unitn.it/). Contact the provider for the paired benchmark; no official direct download is specified here.

Place the MATLAB files in the following layout, or specify `--data-root`.

```
dataset/
  Houston13/  Houston.mat  LiDAR.mat  Houston_GT.mat
  MUUFL/      MUUFL_dataset.mat
  Trento/     HSI_Trento.mat  Lidar_Trento.mat  GT_Trento.mat
```

| Dataset | MATLAB variables | Patch size (formal protocol) |
|---|---|---|
| Houston2013 | img, dsm, Houston_gt | 11 × 11 |
| MUUFL | HSI, LiDAR, Label | 9 × 9 |
| Trento | HSI_Trento, Lidar_Trento, GT_Trento | 9 × 9 |

The fixed train/test indices used by the formal protocol are in [splits/](splits/README.md). Raw sensor scenes are not included.

## Training and evaluation

The formal experiment settings are in `configs/formal.yaml`.

```bash
python formal.py run stage1
python formal.py test tests1
python formal.py summary stage1
```

For the remaining experiments, individual runs, and external comparisons, see [Experiment guide](docs/REPRODUCIBILITY.md).

The code saves generated results and checkpoints locally under `results/`. They are not distributed in this repository.

## Repository layout

| Path | Purpose |
|---|---|
| `symbionet/` | Model, data loading, training, metrics and evaluation |
| `configs/` | Model and experiment settings |
| `formal.py`, `train.py` | Experiment runner and individual training |
| `scripts/` | Robustness evaluation and comparison utilities |
| `splits/`, `masks/` | Fixed split indices and evaluation masks |
| `tests/` | Unit tests |

## Citation and license

See [CITATION.cff](CITATION.cff) for citation information. The software is released under the [MIT License](LICENSE).

We acknowledge the Geo-ΔSSM authors for the fixed split indices and the respective Houston2013, MUUFL and Trento data providers. See [split attribution](splits/README.md).

<!-- Transmission ID: NC-2077 | The signal is still alive. -->
