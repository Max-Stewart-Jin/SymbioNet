# SymbioNet

**Symbiotic Cross-Modal Completion Network for Hyperspectral and LiDAR Joint Classification**

SymbioNet combines gated symmetric fusion, masked labeled counterparts, and spatial-spectral mutual completion (M²C) for HSI–LiDAR classification. The M²C branch is used only during training.

## Installation

Use Python 3.9–3.12 and PyTorch 2.1 or later. For GPU training, install a compatible [CUDA-enabled PyTorch build](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

## Data

Obtain [Houston2013](https://machinelearning.ee.uh.edu/2013-ieee-grss-data-fusion-contest/), [MUUFL](https://github.com/GatorSense/MUUFLGulfport), and [Trento](https://rslab.disi.unitn.it/) from their respective providers under the applicable terms. The paired Trento benchmark may require contacting its provider.

Place the MATLAB files under `dataset/`, or specify `--data-root`.

```text
dataset/
  Houston13/  Houston.mat  LiDAR.mat  Houston_GT.mat
  MUUFL/      MUUFL_dataset.mat
  Trento/     HSI_Trento.mat  Lidar_Trento.mat  GT_Trento.mat
```

| Dataset | MATLAB variables | Patch size |
|---|---|---|
| Houston2013 | `img`, `dsm`, `Houston_gt` | 11 × 11 |
| MUUFL | `HSI`, `LiDAR`, `Label` | 9 × 9 |
| Trento | `HSI_Trento`, `Lidar_Trento`, `GT_Trento` | 9 × 9 |

The fixed split indices in `splits/` are derived from the Geo-ΔSSM benchmark splits; please cite the Geo-ΔSSM work when using them. Raw sensor data are not included.

## Training

```bash
python train.py --dataset houston2013 --exp symbionet --runs 1
```

Use `--dataset muufl` or `--dataset trento` for the other scenes. The Full configuration is defined in `configs/default.yaml` and `configs/datasets.yaml`. Spectral masking is applied only to spatially visible HSI inputs in the unlabeled M²C student, with `spectral_probability=0.5` and a conditional masking rate sampled uniformly from 0 to `spectral_max=0.3`.

Use `--smoke` for a short setup check, `--runs` and `--seed` to control repeated runs, and `--device cpu` for CPU execution. By default, the Full model trains for 500 epochs. The highest clean test OA checkpoint (evaluated every five epochs) and the final-epoch checkpoint are saved locally in `results/<exp>/<dataset>/default/`.

## Evaluation

```bash
python evaluate.py results/symbionet/houston2013/default
```

The evaluator reports OA, AA, Kappa and F1 on clean inputs and under 46 corruption conditions. Corrupted OA is the mean OA over the 46 conditions, with multiple random draws averaged within each condition. For repeated runs, report the mean and population standard deviation across seeds. Use `--weights last` for the final checkpoint or `--suite clean shift interp` for supplementary evaluations. Evaluation outputs are saved locally under `robustness/`.

## Ablation variants

The following labels describe the variants presented in Table III of the manuscript.

| Model | Description |
|---|---|
| (a) | Base network |
| (b) | M²C only |
| (c) | Masked labeled counterparts |
| (d) | Whole-modality masking of labeled counterparts |
| (e) | Masked labeled counterparts and M²C, without the gate |
| (f) | Masked labeled counterparts and the gate, without M²C |
| (g) | SymbioNet with same-position spatial masking |
| Full | SymbioNet with complementary spatial masking |

## License and citation

The software is released under the [MIT License](LICENSE). Third-party data remain subject to their respective terms. Please cite the accompanying SymbioNet manuscript when using this code. Bibliographic details will be added upon publication.
