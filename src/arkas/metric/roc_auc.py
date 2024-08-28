r"""Implement a function to compute the Area Under the Receiver
Operating Characteristic Curve (ROC AUC) metrics."""

from __future__ import annotations

__all__ = [
    "binary_roc_auc_metrics",
    "multiclass_roc_auc_metrics",
    "multilabel_roc_auc_metrics",
    "roc_auc_metrics",
    "preprocess_true_score_binary",
]

from typing import Any

import numpy as np
from sklearn import metrics

from arkas.metric.ap import find_label_type
from arkas.metric.utils import check_label_type, preprocess_true_score_binary


def roc_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the Area Under the Receiver Operating Characteristic Curve
    (ROC AUC) metrics.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        label_type: The type of labels used to evaluate the metrics.
            The valid values are: ``'binary'``, ``'multiclass'``,
            and ``'multilabel'``. If ``'binary'`` or ``'multilabel'``,
            ``y_true`` values  must be ``0`` and ``1``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import roc_auc_metrics
    >>> # auto
    >>> metrics = roc_auc_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ... )
    >>> metrics
    {'count': 5, 'roc_auc': 1.0}
    >>> # binary
    >>> metrics = roc_auc_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_score=np.array([2, -1, 0, 3, 1]),
    ...     label_type="binary",
    ... )
    >>> metrics
    {'count': 5, 'roc_auc': 1.0}
    >>> # multiclass
    >>> metrics = roc_auc_metrics(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_score=np.array(
    ...         [
    ...             [0.7, 0.2, 0.1],
    ...             [0.4, 0.3, 0.3],
    ...             [0.1, 0.8, 0.1],
    ...             [0.2, 0.3, 0.5],
    ...             [0.4, 0.4, 0.2],
    ...             [0.1, 0.2, 0.7],
    ...         ]
    ...     ),
    ...     label_type="multiclass",
    ... )
    >>> metrics
    {'count': 6,
     'macro_roc_auc': 0.833...,
     'micro_roc_auc': 0.826...,
     'roc_auc': array([0.9375, 0.8125, 0.75  ]),
     'weighted_roc_auc': 0.833...}
    >>> # multilabel
    >>> metrics = roc_auc_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
    ...     label_type="multilabel",
    ... )
    >>> metrics
    {'count': 5,
     'macro_roc_auc': 0.666...,
     'micro_roc_auc': 0.544...,
     'roc_auc': array([1., 1., 0.]),
     'weighted_roc_auc': 0.625}

    ```
    """
    check_label_type(label_type)
    if label_type == "auto":
        label_type = find_label_type(y_true=y_true, y_score=y_score)
    if label_type == "binary":
        return binary_roc_auc_metrics(
            y_true=y_true.ravel(), y_score=y_score.ravel(), prefix=prefix, suffix=suffix
        )
    if label_type == "multilabel":
        return multilabel_roc_auc_metrics(
            y_true=y_true, y_score=y_score, prefix=prefix, suffix=suffix
        )
    return multiclass_roc_auc_metrics(y_true=y_true, y_score=y_score, prefix=prefix, suffix=suffix)


def binary_roc_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the Area Under the Receiver Operating Characteristic Curve
    (ROC AUC) metrics for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    y_true, y_score = preprocess_true_score_binary(y_true=y_true, y_score=y_score, nan="remove")

    count = y_true.size
    roc_auc = float("nan")
    if count > 0:
        roc_auc = float(metrics.roc_auc_score(y_true=y_true, y_score=y_score))
    return {f"{prefix}count{suffix}": count, f"{prefix}roc_auc{suffix}": roc_auc}


def multiclass_roc_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the Area Under the Receiver Operating Characteristic Curve
    (ROC AUC) metrics for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    if y_true.shape[0] > 0:
        # Remove NaN values
        mask = np.logical_not(np.logical_or(np.isnan(y_true), np.isnan(y_score).any(axis=1)))
        y_true, y_score = y_true[mask], y_score[mask]

    return _multi_roc_auc_metrics(
        y_true=y_true, y_score=y_score, prefix=prefix, suffix=suffix, multi_class="ovr"
    )


def multilabel_roc_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the Area Under the Receiver Operating Characteristic Curve
    (ROC AUC) metrics for multilabel labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    return _multi_roc_auc_metrics(
        y_true=y_true, y_score=y_score, prefix=prefix, suffix=suffix, multi_class="ovr"
    )


def _multi_roc_auc_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    **kwargs: Any,
) -> dict[str, float | np.ndarray]:
    r"""Return the Area Under the Receiver Operating Characteristic Curve
    (ROC AUC) metrics for multilabel or multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        kwargs: Keyword arguments that are passed to
            ``sklearn.metrics.roc_auc_score``.

    Returns:
        The computed metrics.
    """
    n_samples = y_true.shape[0]
    macro_roc_auc, micro_roc_auc, weighted_roc_auc = [float("nan")] * 3
    n_classes = y_score.shape[1] if y_score.ndim == 2 else 0 if n_samples == 0 else 1
    roc_auc = np.full((n_classes,), fill_value=float("nan"))
    if n_samples > 0:
        macro_roc_auc = float(
            metrics.roc_auc_score(y_true=y_true, y_score=y_score, average="macro", **kwargs)
        )
        micro_roc_auc = float(
            metrics.roc_auc_score(y_true=y_true, y_score=y_score, average="micro", **kwargs)
        )
        weighted_roc_auc = float(
            metrics.roc_auc_score(y_true=y_true, y_score=y_score, average="weighted", **kwargs)
        )
        roc_auc = np.asarray(
            metrics.roc_auc_score(y_true=y_true, y_score=y_score, average=None, **kwargs)
        ).ravel()

    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_roc_auc{suffix}": macro_roc_auc,
        f"{prefix}micro_roc_auc{suffix}": micro_roc_auc,
        f"{prefix}roc_auc{suffix}": roc_auc,
        f"{prefix}weighted_roc_auc{suffix}": weighted_roc_auc,
    }