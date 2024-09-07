r"""Contain functions to compute metrics."""

from __future__ import annotations

__all__ = [
    "accuracy",
    "average_precision",
    "balanced_accuracy",
    "binary_average_precision",
    "binary_confusion_matrix",
    "binary_fbeta_score",
    "binary_jaccard",
    "binary_precision",
    "binary_recall",
    "binary_roc_auc_metrics",
    "confusion_matrix",
    "fbeta_score",
    "jaccard",
    "multiclass_average_precision",
    "multiclass_confusion_matrix",
    "multiclass_fbeta_score",
    "multiclass_jaccard",
    "multiclass_precision",
    "multiclass_recall",
    "multiclass_roc_auc_metrics",
    "multilabel_average_precision",
    "multilabel_confusion_matrix",
    "multilabel_fbeta_score",
    "multilabel_jaccard",
    "multilabel_precision",
    "multilabel_recall",
    "multilabel_roc_auc_metrics",
    "precision",
    "recall",
    "roc_auc_metrics",
    "mean_absolute_error",
    "median_absolute_error",
    "mean_squared_error",
    "mean_squared_log_error",
    "regression_errors",
    "mean_absolute_percentage_error",
]

from arkas.metric.classification.accuracy import accuracy, balanced_accuracy
from arkas.metric.classification.ap import (
    average_precision,
    binary_average_precision,
    multiclass_average_precision,
    multilabel_average_precision,
)
from arkas.metric.classification.confmat import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)
from arkas.metric.classification.fbeta import (
    binary_fbeta_score,
    fbeta_score,
    multiclass_fbeta_score,
    multilabel_fbeta_score,
)
from arkas.metric.classification.jaccard import (
    binary_jaccard,
    jaccard,
    multiclass_jaccard,
    multilabel_jaccard,
)
from arkas.metric.classification.precision import (
    binary_precision,
    multiclass_precision,
    multilabel_precision,
    precision,
)
from arkas.metric.classification.recall import (
    binary_recall,
    multiclass_recall,
    multilabel_recall,
    recall,
)
from arkas.metric.regression.abs_error import mean_absolute_error, median_absolute_error
from arkas.metric.regression.mape import mean_absolute_percentage_error
from arkas.metric.regression.mse import mean_squared_error
from arkas.metric.regression.msle import mean_squared_log_error
from arkas.metric.regression.universal import regression_errors
from arkas.metric.roc_auc import (
    binary_roc_auc_metrics,
    multiclass_roc_auc_metrics,
    multilabel_roc_auc_metrics,
    roc_auc_metrics,
)
