r"""Implement the precision result."""

from __future__ import annotations

__all__ = [
    "binary_precision_metrics",
    "find_label_type",
    "multiclass_precision_metrics",
    "multilabel_precision_metrics",
    "precision_metrics",
]

import math

import numpy as np
from sklearn import metrics

from arkas.metric.utils import check_label_type, preprocess_true_pred


def precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the precision metrics.

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

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import precision_metrics
    >>> # auto
    >>> precision_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'count': 5, 'precision': 1.0}
    >>> # binary
    >>> precision_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     label_type="binary",
    ... )
    {'count': 5, 'precision': 1.0}
    >>> # multiclass
    >>> precision_metrics(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ...     label_type="multiclass",
    ... )
    {'count': 6,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'precision': array([1., 1., 1.]),
     'weighted_precision': 1.0}
    >>> # multilabel
    >>> precision_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     label_type="multilabel",
    ... )
    {'count': 5,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'precision': array([1., 1., 1.]),
     'weighted_precision': 1.0}

    ```
    """
    check_label_type(label_type)
    if label_type == "auto":
        label_type = find_label_type(y_true=y_true, y_pred=y_pred)
    if label_type == "binary":
        return binary_precision_metrics(y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix)
    if label_type == "multilabel":
        return multilabel_precision_metrics(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix
        )
    return multiclass_precision_metrics(y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix)


def binary_precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the precision metrics for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import binary_precision_metrics
    >>> binary_precision_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    {'count': 5, 'precision': 1.0}

    ```
    """
    y_true, y_pred = preprocess_true_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), nan="remove"
    )

    count, precision = y_true.size, float("nan")
    if count > 0:
        precision = float(metrics.precision_score(y_true=y_true, y_pred=y_pred))
    return {f"{prefix}count{suffix}": count, f"{prefix}precision{suffix}": precision}


def multiclass_precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the precision metrics for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multiclass_precision_metrics
    >>> multiclass_precision_metrics(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    ... )
    {'count': 6,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'precision': array([1., 1., 1.]),
     'weighted_precision': 1.0}

    ```
    """
    y_true, y_pred = preprocess_true_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), nan="remove"
    )

    n_samples = y_true.shape[0]
    macro_precision, micro_precision, weighted_precision = float("nan"), float("nan"), float("nan")
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else 0 if n_samples == 0 else 1
    precision = np.full((n_classes,), fill_value=float("nan"))
    if n_samples > 0:
        macro_precision = float(
            metrics.precision_score(
                y_true=y_true, y_pred=y_pred, average="macro", zero_division=0.0
            )
        )
        micro_precision = float(
            metrics.precision_score(
                y_true=y_true, y_pred=y_pred, average="micro", zero_division=0.0
            )
        )
        weighted_precision = float(
            metrics.precision_score(
                y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0.0
            )
        )
        precision = np.asarray(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0.0)
        ).ravel()
    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_precision{suffix}": macro_precision,
        f"{prefix}micro_precision{suffix}": micro_precision,
        f"{prefix}precision{suffix}": precision,
        f"{prefix}weighted_precision{suffix}": weighted_precision,
    }


def multilabel_precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the precision metrics for multilabel labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multilabel_precision_metrics
    >>> multilabel_precision_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ... )
    {'count': 5,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'precision': array([1., 1., 1.]),
     'weighted_precision': 1.0}

    ```
    """
    n_samples = y_true.shape[0]
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else 0 if n_samples == 0 else 1
    precision = np.full((n_classes,), fill_value=float("nan"))
    macro_precision, micro_precision, weighted_precision = [float("nan")] * 3
    if n_samples > 0:
        precision = np.array(
            metrics.precision_score(
                y_true=y_true,
                y_pred=y_pred,
                average="binary" if n_classes == 1 else None,
            )
        ).ravel()
        macro_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        )
        micro_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="micro")
        )
        weighted_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        )
    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_precision{suffix}": macro_precision,
        f"{prefix}micro_precision{suffix}": micro_precision,
        f"{prefix}precision{suffix}": precision,
        f"{prefix}weighted_precision{suffix}": weighted_precision,
    }


def find_label_type(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    r"""Try to find the label type automatically based on the arrays'
    shape and values.

    Note:
        NaN are used to indicate invalid/missing values.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.

    Returns:
        The label type.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric.precision import find_label_type
    >>> # binary
    >>> label_type = find_label_type(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> label_type
    'binary'
    >>> # multiclass
    >>> label_type = find_label_type(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    ... )
    >>> label_type
    'multiclass'

    ```
    """
    # remove NaNs because they indicate missing values
    unique = set(filter(lambda x: not math.isnan(x), np.unique(y_true).tolist()))
    if unique.issubset({0, 1}):
        if y_true.ndim == 2 and y_pred.ndim == 2:
            return "multilabel"
        return "binary"
    return "multiclass"