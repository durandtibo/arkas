from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from arkas.result.binary_classification import compute_binary_metrics


def test_compute_binary_metrics_correct() -> None:
    assert objects_are_equal(
        compute_binary_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "f1": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
        },
        show_difference=True,
    )
    # TODO: test beta
