r"""Contain functions to compute metrics."""

from __future__ import annotations

__all__ = [
    "accuracy_metrics",
    "average_precision_metrics",
    "balanced_accuracy_metrics",
    "fbeta_metrics",
    "jaccard_metrics",
    "precision_metrics",
    "recall_metrics",
    "roc_auc_metrics",
    "confusion_matrix_metrics",
    "binary_confusion_matrix_metrics",
    "multiclass_confusion_matrix_metrics",
]

from arkas.metric.accuracy import accuracy_metrics, balanced_accuracy_metrics
from arkas.metric.ap import average_precision_metrics
from arkas.metric.confmat import (
    binary_confusion_matrix_metrics,
    confusion_matrix_metrics,
    multiclass_confusion_matrix_metrics,
)
from arkas.metric.fbeta import fbeta_metrics
from arkas.metric.jaccard import jaccard_metrics
from arkas.metric.precision import precision_metrics
from arkas.metric.recall import recall_metrics
from arkas.metric.roc_auc import roc_auc_metrics
