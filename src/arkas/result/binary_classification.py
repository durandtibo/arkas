from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn import metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np




def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    f1_betas: Sequence[float] = (1,),
) -> dict[str, float]:
    # TODO: ravel and check inputs
    count = y_true.size
    count_correct = int(metrics.accuracy_score(y_true=y_true, y_pred=y_pred, normalize=False))
    tn, fp, fn, tp = metrics.confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    out = {
        "count": count,
        "count_correct": count_correct,
        "count_incorrect": count - count_correct,
        "accuracy": float(metrics.accuracy_score(y_true=y_true, y_pred=y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred)),
        "precision": float(metrics.precision_score(y_true=y_true, y_pred=y_pred)),
        "recall": float(metrics.recall_score(y_true=y_true, y_pred=y_pred)),
        "jaccard": float(metrics.jaccard_score(y_true=y_true, y_pred=y_pred)),
    }
    out |= {
        f"f{beta}": float(metrics.fbeta_score(y_true=y_true, y_pred=y_pred, beta=beta))
        for beta in f1_betas
    }
    if y_score is not None:
        out |= {
            "average_precision": float(
                metrics.average_precision_score(y_true=y_true, y_score=y_score)
            ),
            "roc_auc": float(
                metrics.roc_auc_score(y_true=y_true, y_score=y_score)
            )
        }
    return out
