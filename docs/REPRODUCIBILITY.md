# Experiment guide

Install dependencies and prepare the datasets following the [README](../README.md) and [Quick start](QUICKSTART.md).

For an environment close to the original runs, use Python 3.10, PyTorch 2.1.0 with CUDA 11.8, NumPy 1.26.4 and SciPy 1.11.4. Numerical results may vary by platform.

## Formal experiments

All commands are run from the repository root using `configs/formal.yaml`.

```bash
python formal.py plan
python formal.py run stage1    # full model and ablations
python formal.py test tests1   # robustness and supplementary evaluations
python formal.py run stage3    # fewer labeled pixels
python formal.py test tests2   # robustness of fewer-label models
python formal.py run stage4    # parameter analysis
python formal.py run stage5    # unlabeled pool size
python formal.py timing
python formal.py maps
python formal.py summary
python formal.py pack
```

Use `--data-root DIR`, `--device cuda|cpu`, `--seeds 0,1`, `--limit N` or `--smoke` as needed. Finished runs and tests are skipped.

Use the fixed `*_geodeltassm.npz` splits, not the alternative random splits, for the formal protocol. Inputs are normalized per band; patches use the dataset-specific settings. The primary training protocol uses 500 epochs and five seeds, with three seeds for parameter analysis. Best checkpoints are selected using clean test OA evaluated every five epochs; final-epoch checkpoints are also saved.

Robustness is measured over 46 conditions: Gaussian noise, pixel loss, block occlusion, band loss and missing modality. Random draws are averaged within each condition before averaging the conditions and seeds.

## Individual models

```bash
python train.py --dataset houston2013 --exp demo --runs 1 --set data.patch=11 eval.every=5 model.decoder=identity model.mask_labeled=true model.pre_gate=known model.pre_gate_p=0.5 model.rec_weighting=none model.lam=0 model.mask_mode=complementary model.fill=mean
python -m scripts.robustness --help
```

For MUUFL and Trento, use `--dataset muufl` or `--dataset trento` with `data.patch=9`. The options and defaults are in `configs/default.yaml`.

## External comparisons

External implementations must be obtained from their respective authors.

| Method | Repository | Required source path |
|---|---|---|
| DAHGM | [DAHGMN](https://github.com/RogsXie/DAHGMN) | model/DAHGM.py |
| CNN_H | [Coupled-CNNs](https://github.com/RenlongHang/Coupled-CNNs) | CoupledCNNs.py |
| EndNet | [Multimodal-Remote-Sensing-Toolkit](https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit) | model/EndNet.py |
| MFT | [MFT](https://github.com/AnkurDeria/MFT) | Model/MFT.ipynb |

Place each implementation in a subdirectory of `external/`, preserving its source layout.

```bash
python -m scripts.compare check
python -m scripts.compare all
```

For the CNN_H masked-copy transfer experiment:

```bash
python -m scripts.compare all --methods cnnh
python -m scripts.cnn_copies check
python -m scripts.cnn_copies all
```

## Local outputs and checks

Training and evaluation save run records, checkpoints, configuration files, test scores and logs under `results/`. `formal.py pack` archives result records without checkpoints. Generated files are excluded from version control.

```bash
python tests/test_model.py
python tests/test_data.py
python tests/test_corrupt.py
python tests/test_conditions.py
python tests/test_testing.py
```
