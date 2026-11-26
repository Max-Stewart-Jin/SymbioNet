# Auxiliary experiment scripts

Run commands from the **repository root** with `python -m scripts.<module>`.
The primary paper protocol remains `python formal.py ...` and single-model training remains `python train.py ...`.

| Module | Purpose | Example |
|---|---|---|
| `compare` | External model comparison | `python -m scripts.compare check` |
| `cnn_copies` | CNN_H masked-copy transfer | `python -m scripts.cnn_copies check` |
| `robustness` | Robustness scoring for individual runs | `python -m scripts.robustness --help` |
| `completion_check` | Cross-modal completion diagnostic | `python -m scripts.completion_check --help` |
| `cnn_copies_train`, `cnn_copies_eval` | Support modules for CNN_H transfer | Imported by `cnn_copies` |

See [Experiment guide](../docs/REPRODUCIBILITY.md) for the full protocol.
