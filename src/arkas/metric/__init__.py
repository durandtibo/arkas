r"""Contain functions to compute metrics."""

from __future__ import annotations

__all__ = [
    "accuracy_metrics",
    "average_precision_metrics",
    "balanced_accuracy_metrics",
    "binary_confusion_matrix_metrics",
    "binary_precision_metrics",
    "binary_recall_metrics",
    "confusion_matrix_metrics",
    "fbeta_metrics",
    "jaccard_metrics",
    "multiclass_confusion_matrix_metrics",
    "multiclass_precision_metrics",
    "multiclass_recall_metrics",
    "multilabel_precision_metrics",
    "multilabel_recall_metrics",
    "precision_metrics",
    "recall_metrics",
    "roc_auc_metrics",
    "binary_jaccard_metrics",
    "multiclass_jaccard_metrics",
    "multilabel_jaccard_metrics",
]

from arkas.metric.accuracy import accuracy_metrics, balanced_accuracy_metrics
from arkas.metric.ap import average_precision_metrics
from arkas.metric.confmat import (
    binary_confusion_matrix_metrics,
    confusion_matrix_metrics,
    multiclass_confusion_matrix_metrics,
)
from arkas.metric.fbeta import fbeta_metrics
from arkas.metric.jaccard import (
    binary_jaccard_metrics,
    jaccard_metrics,
    multiclass_jaccard_metrics,
    multilabel_jaccard_metrics,
)
from arkas.metric.precision import (
    binary_precision_metrics,
    multiclass_precision_metrics,
    multilabel_precision_metrics,
    precision_metrics,
)
from arkas.metric.recall import (
    binary_recall_metrics,
    multiclass_recall_metrics,
    multilabel_recall_metrics,
    recall_metrics,
)
from arkas.metric.roc_auc import roc_auc_metrics
