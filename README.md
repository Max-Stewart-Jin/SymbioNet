# SymbioNet

Code of the paper **SymbioNet: Symbiotic Cross-Modal Completion Network for Hyperspectral and LiDAR Joint
Classification**.

SymbioNet classifies every pixel of a co-registered hyperspectral image (HSI) and LiDAR raster from very few labels
(1 % of the labeled pixels of each class). It is trained in a single stage for clean and for degraded inputs: noise,
lost pixels, occluded blocks, lost HSI bands, or a missing modality.

- **Gated symmetric fusion block (GSFB).** Each modality has its own projections, and its queries attend to the
  tokens of both modalities. A gate without parameters finds the positions whose data are lost and removes them from
  the attention keys. These positions keep their queries and are updated from the available positions of both
  modalities.
- **Masked labeled copies.** The classifier is also trained on masked copies of the labeled patches.
- **Masked mutual completion (M2C).** Patches from the unlabeled pool are masked complementarily across the two
  modalities, so every masked position stays visible in the other modality. Linear heads predict, at the masked
  positions, the features that the same encoder computes from the unmasked patch. The branch shares the encoder with
  the classifier and is removed after training.

The network used at test time is the fusion encoder (two GSFBs) with a linear classifier. It has 0.23 to 0.28 M
parameters, depending on the dataset.

## Getting started

New to the repository? See **[Quick start](docs/QUICKSTART.md)** for Windows/Linux setup,
source integrity checks, GPU detection, and an optional smoke test.

No need to repeat the historical study merely to verify the repository layout: run
`python tools/verify_release.py` and `python formal.py plan` after installing dependencies.
These checks do **not** claim a full numerical reproduction.

## Requirements

Python 3.9 to 3.12, PyTorch 2.1 or later, and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

No compiled extension is needed. The experiments of the paper were run on Windows with a laptop GPU (RTX 4060,
8 GB); the unit tests also pass on Linux.

## Data

The scenes are not part of this repository; they are obtained from their providers (see Acknowledgment;
Houston2013 is distributed by the University of Houston under its terms of use). Put them in `dataset/` next to
this file, or pass `--data-root`:

```
dataset/
  Trento/      HSI_Trento.mat   Lidar_Trento.mat   GT_Trento.mat
  Houston13/   Houston.mat      LiDAR.mat          Houston_GT.mat
  MUUFL/       MUUFL_dataset.mat
```

| dataset | variables read from the files | scene | classes | training / test pixels | patch |
|---|---|---|---|---|---|
| Trento | `HSI_Trento`, `Lidar_Trento`, `GT_Trento` | 166 x 600 | 6 | 306 / 29908 | 9 x 9 |
| Houston2013 | `img`, `dsm`, `Houston_gt` | 349 x 1905 | 15 | 158 / 14871 | 11 x 11 |
| MUUFL | `HSI`, `LiDAR`, `Label` (unlabeled pixels are -1) | 325 x 220 | 11 | 541 / 53146 | 9 x 9 |

**Splits.** The training and test pixels are those of Geo-ΔSSM (`splits/<dataset>_geodeltassm.npz`, provided by its
authors): for each class, ceil(1 % x class size) training pixels, and all other labeled pixels for testing.
`splits/<dataset>_ratio0p01_seed0.npz` are random splits of the same sizes (`data.split: random`).

**Preprocessing.** Per-band min-max scaling of the HSI (all bands) and of the LiDAR over the whole image, zero
padding outside the image, random flips of the training patches.

**Unlabeled pool.** Every pixel of the scene that has data, except the test pixels. Only the inputs of the pool
pixels are used, never their labels. The test pixels are never drawn for training; as in any patch-based method,
their spectra can appear inside the patches of neighbouring pixels.

## Reproducing the paper

The authors' full historical results are retained separately and are not part of this initial source-only
commit. See [reference values](docs/RESULTS_REFERENCE.md) and [source provenance](docs/SOURCE_PROVENANCE.md).
Running the entire training protocol is for independent reproduction and is not required for reading this repository.

All experiments are defined in `configs/formal.yaml` and run by `formal.py`:

```bash
python formal.py plan               # what is planned, what is on disk, an estimate of the remaining time
python formal.py run stage1         # comparison and ablation: 9 versions x 3 datasets x 5 seeds
python formal.py test tests1        # the stage1 models under the test conditions (no training)
python formal.py run stage3         # one half and one quarter of the training pixels
python formal.py run stage4         # parameter analysis (3 seeds)
python formal.py run stage5         # size of the unlabeled pool (Houston2013)
python formal.py test tests2        # the stage3 models under the 46 corrupted conditions
python formal.py timing             # seconds per training epoch, milliseconds per test patch
python formal.py maps               # the predicted class of every pixel, for the classification maps
python formal.py summary            # the tables of what is on disk (or: summary stage1)
python formal.py pack               # results/SymbioNet_formal_results.zip: every record, without the weights
```

A run is one seed of one version on one dataset. Finished runs and finished test conditions are skipped, so every
command can be stopped and started again. A test stage needs the runs of its training stage (`tests1` those of
`stage1`, `tests2` those of `stage3`). Options: `--data-root DIR`, `--device cuda|cpu`, `--seeds 0,1` (train only
these seeds), `--limit N` (train at most N runs), and `--smoke` (2 epochs on part of the test set, saved under
`results/formal_smoke`; use it to check the setup before the long runs).

Versions of `stage1`:

| version | what is trained |
|---|---|
| `full` | SymbioNet: masked labeled copies + M2C + the gate |
| `base` | the network alone: clean labeled patches, no masked view, no gate |
| `m2c` | base + M2C |
| `copies` | base + masked labeled copies |
| `copies_whole` | as `copies`, but a copy loses a whole modality instead of pixels (modality dropout) |
| `copies_m2c` | masked labeled copies + M2C, without the gate |
| `copies_gate` | masked labeled copies + the gate, without M2C |
| `full_indep` | the full method with independent masks in the two modalities |
| `full_same` | the full method with the same positions masked in both modalities |

The versions of the other stages (`full_half`, `full_depth1`, `full_pool18k`, ...) are `full`, `base` or `copies`
with changed settings; `configs/formal.yaml` lists them with comments.

**Training.** 500 epochs, Adam (learning rate 0.001, weight decay 0.001, cosine annealing), cross-entropy with label
smoothing 0.2 and class weights proportional to 1 / sqrt(training pixels of the class), batches of 32 labeled
patches (MUUFL: 128) and 256 pool patches. The test pixels are scored every 5 epochs; the
scores at the epoch with the best OA (`best`) and at the last epoch (`last`) are both saved. Every run is seeded
(run k uses seed k). The numbers can differ slightly between GPUs and PyTorch versions.

**Test conditions** (`symbionet/corrupt.py`, `symbionet/conditions.py`). A condition changes the preprocessed image
before the patches are cut, lost values are replaced by the band means of the image, and all models are scored on
the same images. The levels of a test are nested: what is lost at one level is also lost at every higher level.

| test | conditions | number |
|---|---|---|
| noise | Gaussian noise at an SNR of 30, 20, 10, 5 and 0 dB per band, on the HSI, the LiDAR or both | 15 |
| lost pixels | 10, 30, 50, 70 and 90 % of the pixels, in the HSI, the LiDAR or both | 15 |
| occluded blocks | 16 x 16 blocks over 10 to 50 % of the image, in the HSI or the LiDAR (mean of 3 draws) | 10 |
| lost bands | 10, 30, 50 and 70 % of the HSI bands (mean of 5 draws) | 4 |
| missing modality | the HSI or the LiDAR is missing | 2 |

`tests1` adds three further tests: the LiDAR raster shifted by 1, 2 and 3 pixels against the HSI (mean
of the four directions), lost LiDAR values filled by nearest-neighbour interpolation instead of the band means, and
the accuracy inside and outside the cloud shadow of Houston2013 (`masks/README.md`).

**Outputs**, under `results/formal/`:

| file | content |
|---|---|
| `<dataset>/<version>/run<seed>.json` | OA, AA, kappa, F1, precision, per-class accuracy and confusion matrix at the best and at the last epoch; training curves; time; number of parameters |
| `<dataset>/<version>/ckpt/run<seed>_best.pt`, `run<seed>_last.pt` | the weights |
| `<dataset>/<version>/summary.json`, `config.yaml` | mean and standard deviation over the seeds; the resolved settings |
| `tests/<dataset>.json` | the scores of every tested model, by version, condition and seed |
| `timing.json`, `maps/` | results of `timing` and `maps` |
| `log.txt` | everything printed by `run` and `test` |

## Training one model

`train.py` trains any setting of `configs/default.yaml` (all options, with comments), with the per-dataset values
of `configs/datasets.yaml` and the overrides given with `--set`. Without overrides it trains the base network. The
settings of `full` on Houston2013 are:

```bash
python train.py --dataset houston2013 --exp demo --runs 1 --set data.patch=11 eval.every=5 model.decoder=identity model.mask_labeled=true model.pre_gate=known model.pre_gate_p=0.5 model.rec_weighting=none model.lam=0 model.mask_mode=complementary model.fill=mean
```

Trento and MUUFL use `data.patch=9`. `--smoke` runs 2 epochs on part of the test set, and `--sweep key=v1,v2` runs
every combination of the listed values. The results are saved under `results/<exp>/<dataset>/<setting>/` in the
form described above.

How the terms of the paper map to the settings:

| paper | settings |
|---|---|
| GSFB | the class `SFB` in `symbionet/model.py`; `model.depth` blocks of width `model.d` |
| gate, and the share of masked patches it acts on in training | `model.pre_gate: known`, `model.pre_gate_p` |
| masked labeled copies and their weight | `model.mask_labeled: true`, `training.masked_weight` |
| M2C and the weight of its feature loss | `model.decoder: identity` (with `model.lam: 0`, `model.rec_weighting: none`), `model.beta` |
| complementary masks, mask ratio | `model.mask_mode: complementary`, `model.mask_ratio` |
| masked values set to the band means | `model.fill: mean` |
| pool patches per step, size of the pool | `training.pool_batch`, `data.pool_size` |

`configs/default.yaml` and `symbionet/model.py` also contain options that were tried during development and are not
used in the paper (other decoders, a pixel reconstruction loss, other stems and readouts).

`robustness.py` scores runs of `train.py` under the 46 corrupted conditions:

```bash
python robustness.py results/demo/houston2013/default --names SymbioNet
```

## Tests

```bash
python tests/test_model.py        # the network, the masks, the losses and their gradients (random data)
python tests/test_data.py         # splits, preprocessing, patch sampling (uses dataset/ if it is there)
python tests/test_corrupt.py      # the corruptions
python tests/test_conditions.py   # the shifted raster, the interpolation, the suites of conditions
python tests/test_testing.py      # the test stages and their score files
python tests/test_formal.py       # the plan of the experiments
```

## Repository layout

```
symbionet/
  model.py        the network: stems, GSFBs, classifier, masks, the M2C branch and its losses
  data.py         scenes, preprocessing, splits, patch sampling, the unlabeled pool
  engine.py       one training run
  runs.py         training runs on disk
  corrupt.py      the corruptions of the robustness tests
  conditions.py   further test conditions and the suites of conditions
  testing.py      trained models scored under test conditions
  metrics.py      OA, AA, kappa, F1, precision, per-class accuracy
  config.py       settings
configs/          default.yaml (all options), datasets.yaml, formal.yaml (the experiments of the paper)
splits/           training and test pixels
masks/            the cloud-shadow zone map of Houston2013
formal.py         the experiments of the paper
train.py          training of any setting
robustness.py     corruption tests for runs of train.py
tests/            unit tests
```

## Citation

The reference of the paper will be added here when it is published.

## Acknowledgment

The splits are those of Geo-ΔSSM. The Trento data were provided by L. Bruzzone (University of Trento). The
Houston2013 data come from the 2013 IEEE GRSS Data Fusion Contest (Hyperspectral Image Analysis group and NCALM,
University of Houston). The MUUFL Gulfport data were collected by P. Gader, A. Zare and their colleagues.

## License

MIT (see `LICENSE`).

<!-- Transmission ID: NC-2077 | The signal is still alive. -->
