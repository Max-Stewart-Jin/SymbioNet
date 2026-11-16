# Reproducing the SymbioNet experiments

This document describes the protocol of the manuscript and the additional CNN$_H$ transfer comparison.
The code reproduces the experiments using **three datasets and five training seeds** in the primary protocol.
Parameter sweeps use three seeds per configuration. The numerical values depend on the dataset files, GPU,
PyTorch build, and the original source revisions of external models.

## 1. Environment and sensor data

```bash
python -m pip install -r requirements.txt
```

Use Python 3.9–3.12 and a CUDA-enabled PyTorch build for practical training times. The original runs used an
NVIDIA RTX 4060 laptop GPU. Put the original MATLAB data in `dataset/` with the exact layout described in the
main README, including [the data acquisition links](../README.md#data). The raw scenes are not redistributed here;
obtain them from the respective data providers under their terms.

### Recorded experiment environment (historical reference)

| Component | Version / hardware | Evidence |
|---|---|---|
| Python | 3.10.20 | Workstation `hsi` environment notes |
| PyTorch | 2.1.0+cu118 (CUDA 11.8 build) | Original per-run JSON metadata (430 records in the retained formal result archive) |
| NumPy | 1.26.4 | Workstation `hsi` environment notes |
| SciPy | 1.11.4 | Workstation `hsi` environment notes |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB | Original run metadata and project notes |

For researchers who want to approximate that original **Windows/Linux CUDA 11.8** environment, use Python 3.10
and install the project dependencies, then select the archived numerical-library versions:

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 scipy==1.11.4
python -c "import sys, torch, numpy, scipy; print(sys.version, torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__); print('GPU:', torch.cuda.is_available())"
```

See the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/)
for the wheel and supported platform. The above setup is **optional**, not required for the no-training
integrity check, and has not been independently full-GPU-retested from this release. The historical archives
do **not** include a complete `pip freeze`/lockfile (for example, exact PyYAML and h5py versions are not
confirmed); `requirements.txt` intentionally keeps compatibility minimums. No claim of exact numerical
reproduction across machines is implied.

The `.npz` files in `splits/` contain **indices**, not hyperspectral or LiDAR sensor arrays.
The paper uses the Geo-ΔSSM fixed splits. **The three `*_geodeltassm.npz` index files are included**
with reuse permission confirmed by a collaborating author. They were reconstructed from the Geo-ΔSSM
patch files by matching training patch centres; see [`splits/README.md`](../splits/README.md).
Do not substitute the provided `ratio0p01_seed0` random splits when reproducing the manuscript.

The original per-seed results are retained by the authors in the separately archived formal and CNN_H results packages.
They are not part of this initial source-code commit. New training runs should create their own `results/` content and checkpoints.

For a fast consistency check without dataset files:

```bash
python tests/test_conditions.py
python tests/test_corrupt.py
python tests/test_model.py
python tests/test_testing.py
python formal.py plan
```

`tests/test_formal.py` performs a miniature training pass for many configurations and can be slow without a GPU.
External-model tests may skip checks when the published source trees are absent.

## 2. Training SymbioNet and internal ablations

All commands below use `configs/formal.yaml`, including the model settings, the masks, the same labeled split,
and the train/test conditions. The results go to `results/formal/`. All stages can resume finished runs/tests.

```bash
python formal.py plan
python formal.py run stage1    # 9 versions × 3 datasets × 5 seeds = 135 trainings
python formal.py test tests1   # 46 corrupted conditions and planned supplementary tests
python formal.py run stage3    # 6 fewer-label versions × 3 datasets × 5 seeds = 90 trainings
python formal.py test tests2   # 46 conditions for all 90 fewer-label runs
python formal.py run stage4    # 15 parameter configurations × 3 datasets × 3 seeds = 135 trainings
python formal.py run stage5    # 2 unlabeled pool sizes × 1 dataset × 5 seeds = 10 trainings
python formal.py timing
python formal.py maps
python formal.py summary
python formal.py pack
```

`tests1` contains best-epoch robustness, additional diagnostic tests and the preconfigured **last-epoch** checks.
Scores under a condition with several random realizations are first averaged across those realizations.
The 46-condition robustness score is then averaged over conditions *within each seed*, followed by averaging
across the five seeds. This is not the mean of all individual noisy draws with unequal weights.

**Model selection disclosure:** The historical experiment protocol selects the checkpoint with the highest
**clean test-set OA** evaluated every five epochs, rather than holding out a separate validation set. It also
reports the final-epoch checkpoints, without this epoch selection. Results should be described with this
limitation; do not present the selected checkpoint as an independently validation-selected estimate.

## 3. Reproducing comparisons with published networks

External model implementations are not redistributed here. Obtain the public versions from their authors:

| Method | Source repository | Required source path in repository |
|---|---|---|
| DAHGM | https://github.com/RogsXie/DAHGMN | `model/DAHGM.py` |
| CNN$_H$ | https://github.com/RenlongHang/Coupled-CNNs | `CoupledCNNs.py` |
| EndNet | https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit | `model/EndNet.py` |
| MFT | https://github.com/AnkurDeria/MFT | `Model/MFT.ipynb` |

Place each repository in any subfolder of `external/`, preserving its source file's relative path. Reference
source SHA-256 values are in `symbionet/external.py`. Other source revisions may have different results.

```bash
python -m scripts.compare check
python -m scripts.compare all
```

The external-model adapter shares the experimental split, normalization of the scene, corruption images, and
base training protocol, but preserves the individual model's special preprocessing, loss, and prediction rule.
It does **not** imply identical architectures or training objectives.

The relocated `scripts.compare` command may wait for **1500 seconds of log inactivity** before acquiring the formal
experiment queue when used with its default wait mode. To run immediately only if no other experiment currently
holds the queue, it also offers `--no-wait`. This safety delay does not change model training or evaluation.

## 4. CNN$_H$ masked labeled copies

Train and score the original CNN$_H$ baseline from section 3 first; the transfer driver verifies that the fixed
splits, training settings, and CNN$_H$ source hashes match the reference run records before it begins.

```bash
python -m scripts.compare all --methods cnnh
python -m scripts.cnn_copies check
python -m scripts.cnn_copies smoke      # quick sample, if desired
python -m scripts.cnn_copies all        # 3 datasets × 5 seeds = 15 new trainings, 46 corruptions + clean per seed
```

`cnn_copies.py all` also uses the original queue safety delay. The new CNN$_H$ runs start from random initialization
and share their seed index with their corresponding original-CNN$_H$ runs. The extra objective is the supervised
classification loss on a masked labeled copy. This comparison measures a *training-strategy transfer across two
architectures*, not an isolated effect of GSFB or the masked completion head.

The paired results are saved under `results/cnn_copies/`, with a packaged JSON summary at
`results/SymbioNet_cnn_copies_results.zip` after completion. Do not compare the CNN$_H$ modified model to the
unmodified literature numbers as though they used the same training recipe.

## 5. Releasing results and trained weights

The repository's `.gitignore` excludes sensor scenes, generated result files, and model weights. `formal.py pack`
collects **results records, not checkpoints**. If sharing pretrained checkpoints or full per-seed results, publish
them separately only after verifying dataset license, authorship, and the absence of personal file paths in the
archives. The commands above are the primary reproduction path.
