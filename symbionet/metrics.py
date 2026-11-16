"""Classification scores from a confusion matrix (rows = true class, columns = prediction), all in percent:
overall accuracy (OA), average per-class accuracy (AA), Cohen's kappa, macro precision and macro F1 (a class that is
never predicted counts as precision 0, as scikit-learn with zero_division=0), and the per-class accuracies."""
import numpy as np
import torch

SCORES = ('OA', 'AA', 'Kappa', 'F1', 'Precision')


def confusion(pred: torch.Tensor, target: torch.Tensor, n_classes: int) -> np.ndarray:
    k = n_classes
    return torch.bincount(target * k + pred, minlength=k * k).reshape(k, k).cpu().numpy()


def scores(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=np.float64)
    n, tp = cm.sum(), np.diag(cm)
    true, predicted = cm.sum(1), cm.sum(0)
    per_class = tp / np.maximum(true, 1)
    oa = tp.sum() / n
    pe = float((true * predicted).sum()) / n ** 2
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 1.0
    return dict(OA=100 * oa, AA=100 * per_class.mean(), Kappa=100 * kappa,
                F1=100 * (2 * tp / np.maximum(true + predicted, 1)).mean(),
                Precision=100 * (tp / np.maximum(predicted, 1)).mean(),
                per_class=(100 * per_class).tolist())
