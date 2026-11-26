# Quick start

Run the commands from the repository root. Python 3.9–3.12 is supported.

## Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
.\.venv\Scripts\python.exe formal.py plan
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
python formal.py plan
```

If CUDA is unavailable, install a compatible PyTorch build using the [official instructions](https://pytorch.org/get-started/locally/). CPU execution is suitable for initial checks.

## Check and run

Prepare the datasets as described in the [README](../README.md#data). Use the fixed splits in `splits/` for the formal experiments.

After activating your environment, run:

```bash
python formal.py run stage1 --smoke --limit 1
python formal.py run stage1
python formal.py test tests1
```

On Windows, replace `python` with `.\.venv\Scripts\python.exe` if the environment is not activated.

The smoke test uses two epochs and a test subset, saved separately in `results/formal_smoke/`. Full runs are saved under `results/formal/` and can resume completed runs.

See the [experiment guide](REPRODUCIBILITY.md) for additional commands.
