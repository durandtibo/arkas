r"""Implement the recall metrics."""

from __future__ import annotations

__all__ = [
    "binary_recall",
    "multiclass_recall",
    "multilabel_recall",
    "recall",
]


import numpy as np
from sklearn import metrics

from arkas.metric.classification.precision import find_label_type
from arkas.metric.utils import (
    check_label_type,
    preprocess_pred,
    preprocess_pred_multilabel,
)


def recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Return the recall metrics.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must be an array of
            shape ``(n_samples,)`` or ``(n_samples, n_classes)``.
        label_type: The type of labels used to evaluate the metrics.
            The valid values are: ``'binary'``, ``'multiclass'``,
            and ``'multilabel'``. If ``'binary'`` or ``'multilabel'``,
            ``y_true`` values  must be ``0`` and ``1``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import recall
    >>> # auto
    >>> recall(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'count': 5, 'recall': 1.0}
    >>> # binary
    >>> recall(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     label_type="binary",
    ... )
    {'count': 5, 'recall': 1.0}
    >>> # multiclass
    >>> recall(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ...     label_type="multiclass",
    ... )
    {'count': 6,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}
    >>> # multilabel
    >>> recall(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     label_type="multilabel",
    ... )
    {'count': 5,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}

    ```
    """
    check_label_type(label_type)
    if label_type == "auto":
        label_type = find_label_type(y_true=y_true, y_pred=y_pred)
    if label_type == "binary":
        return binary_recall(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix, ignore_nan=ignore_nan
        )
    if label_type == "multilabel":
        return multilabel_recall(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix, ignore_nan=ignore_nan
        )
    return multiclass_recall(
        y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix, ignore_nan=ignore_nan
    )


def binary_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float]:
    r"""Return the recall metrics for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import binary_recall
    >>> binary_recall(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'count': 5, 'recall': 1.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=ignore_nan
    )

    count, score = y_true.size, float("nan")
    if count > 0:
        score = float(metrics.recall_score(y_true=y_true, y_pred=y_pred))
    return {f"{prefix}count{suffix}": count, f"{prefix}recall{suffix}": score}


def multiclass_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Return the recall metrics for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multiclass_recall
    >>> multiclass_recall(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    ... )
    {'count': 6,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=ignore_nan
    )

    per_class = np.array([])
    macro, micro, weighted = float("nan"), float("nan"), float("nan")
    n_samples = y_true.shape[0]
    if n_samples > 0:
        macro = metrics.recall_score(
            y_true=y_true, y_pred=y_pred, average="macro", zero_division=0.0
        )
        micro = metrics.recall_score(
            y_true=y_true, y_pred=y_pred, average="micro", zero_division=0.0
        )
        weighted = metrics.recall_score(
            y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0.0
        )
        per_class = np.asarray(
            metrics.recall_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0.0)
        ).ravel()
    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_recall{suffix}": float(macro),
        f"{prefix}micro_recall{suffix}": float(micro),
        f"{prefix}recall{suffix}": per_class,
        f"{prefix}weighted_recall{suffix}": float(weighted),
    }


def multilabel_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Return the recall metrics for multilabel labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multilabel_recall
    >>> multilabel_recall(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ... )
    {'count': 5,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}

    ```
    """
    y_true, y_pred = preprocess_pred_multilabel(y_true, y_pred, remove_nan=ignore_nan)

    per_class = np.array([])
    macro, micro, weighted = float("nan"), float("nan"), float("nan")
    n_samples = y_true.shape[0]
    if n_samples > 0:
        per_class = np.array(
            metrics.recall_score(
                y_true=y_true,
                y_pred=y_pred,
                average="binary" if y_pred.shape[1] == 1 else None,
            )
        ).ravel()
        macro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="macro")
        micro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="micro")
        weighted = metrics.recall_score(y_true=y_true, y_pred=y_pred, average="weighted")

    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_recall{suffix}": float(macro),
        f"{prefix}micro_recall{suffix}": float(micro),
        f"{prefix}recall{suffix}": per_class,
        f"{prefix}weighted_recall{suffix}": float(weighted),
    }
