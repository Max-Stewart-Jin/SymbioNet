# Quick start: SymbioNet reproduction environment

Use this guide if you want to **run or inspect the released research code**. The original training results
already exist in the authors' archives; this guide does **not** ask the original authors to retrain them.
Full numerical reproduction is optional for readers and requires the original sensor datasets and a GPU.

## 1. Windows (PowerShell)

From the project directory, with Python 3.11 installed:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools/verify_release.py
.\.venv\Scripts\python.exe formal.py plan
```

The first two checks require **no HSI/LiDAR dataset** and do **not** train a model. In particular,
`verify_release.py` checks the historical source-file SHA-256 values and the fixed split indices.
`formal.py plan` reads the experiment configuration and shows planned runs.

To train on a GPU, check that your installed PyTorch is CUDA-enabled:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the result is `False`, install a PyTorch build compatible with your NVIDIA driver using the
[official PyTorch selector](https://pytorch.org/get-started/locally/) and re-check. An installed CUDA toolkit
alone does not ensure that Python's PyTorch can use a GPU.

## 2. Linux / macOS

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python tools/verify_release.py
python formal.py plan
```

CUDA acceleration requires a compatible NVIDIA GPU and CUDA-enabled PyTorch (generally Linux or Windows).
On macOS, use CPU for initial checks; full training would be very slow.

## 3. Dataset layout and optional smoke test

Sensor scenes are **not redistributed**. See the main [README](../README.md#data) for names and expected `.mat`
variables. The three **fixed** Geo-ΔSSM training/test index files are included in `splits/`. Do not use the
`*_ratio0p01_seed0.npz` random alternatives for the manuscript numbers.

When the datasets have been obtained legitimately, place them in `dataset/` or provide `--data-root`.
For example, on Windows:

```powershell
.\.venv\Scripts\python.exe formal.py run stage1 --smoke --limit 1 --device cuda --data-root "G:\SymbioNet\dataset"
```

This smoke test runs **two epochs**, writes into `results/formal_smoke/`, and verifies that the training path
starts; its OA is **not** a publication result. It does not alter archived historical runs unless you deliberately
point it to their result directory. To run the full protocol, see [REPRODUCIBILITY](REPRODUCIBILITY.md).

## 4. What has / has not been verified

- The source snapshot and fixed split indices can be verified without training using `tools/verify_release.py`.
- The lightweight CPU unit tests can be run manually or by GitHub Actions.
- Published results were produced earlier on the authors' machine; the public source commit has **not** undergone
  an independent full GPU re-run.
- The historical protocol selects the checkpoint with highest **clean test-set OA**, and records final-epoch
  scores as an additional comparison. This is not independent validation-based checkpoint selection.
- Third-party model source code is **not redistributed**; follow `docs/REPRODUCIBILITY.md` for external comparisons.
