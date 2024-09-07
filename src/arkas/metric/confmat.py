r"""Contain confusion matrix metrics."""

from __future__ import annotations

__all__ = [
    "binary_confusion_matrix_metrics",
    "confusion_matrix_metrics",
    "multiclass_confusion_matrix_metrics",
    "multilabel_confusion_matrix_metrics",
]


import numpy as np
from sklearn import metrics

from arkas.metric.classification.precision import find_label_type
from arkas.metric.utils import (
    check_label_type,
    preprocess_pred,
    preprocess_pred_multilabel,
)


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the confusion matrix metrics.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
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
    >>> from arkas.metric import confusion_matrix_metrics
    >>> # binary
    >>> confusion_matrix_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     label_type="binary",
    ... )
    {'confusion_matrix': array([[2, 0], [0, 3]]),
     'count': 5,
     'false_negative_rate': 0.0,
     'false_negative': 0,
     'false_positive_rate': 0.0,
     'false_positive': 0,
     'true_negative_rate': 1.0,
     'true_negative': 2,
     'true_positive_rate': 1.0,
     'true_positive': 3}
    >>> # multiclass
    >>> confusion_matrix_metrics(
    ...     y_true=np.array([0, 1, 1, 2, 2, 2]),
    ...     y_pred=np.array([0, 1, 1, 2, 2, 2]),
    ...     label_type="multiclass",
    ... )
    {'confusion_matrix': array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]), 'count': 6}
    >>> # multilabel
    >>> confusion_matrix_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     label_type="multilabel",
    ... )
    {'confusion_matrix': array([[[2, 0], [0, 3]],
                                [[3, 0], [0, 2]],
                                [[2, 0], [0, 3]]]),
     'count': 5}

    ```
    """
    check_label_type(label_type)
    if label_type == "auto":
        label_type = find_label_type(y_true=y_true, y_pred=y_pred)
    if label_type == "binary":
        return binary_confusion_matrix_metrics(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix
        )
    if label_type == "multilabel":
        return multilabel_confusion_matrix_metrics(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix
        )
    return multiclass_confusion_matrix_metrics(
        y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix
    )


def binary_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the confusion matrix metrics for binary labels.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import binary_confusion_matrix_metrics
    >>> binary_confusion_matrix_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    {'confusion_matrix': array([[2, 0], [0, 3]]),
     'count': 5,
     'false_negative_rate': 0.0,
     'false_negative': 0,
     'false_positive_rate': 0.0,
     'false_positive': 0,
     'true_negative_rate': 1.0,
     'true_negative': 2,
     'true_positive_rate': 1.0,
     'true_positive': 3}

    ```
    """
    y_true, y_pred = preprocess_pred(y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=True)

    count = y_true.size
    if count > 0:
        confmat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    else:
        confmat = np.zeros((2, 2), dtype=np.int64)
    tn, fp, fn, tp = confmat.ravel().tolist()
    neg = tn + fp
    pos = tp + fn
    return {
        f"{prefix}confusion_matrix{suffix}": confmat,
        f"{prefix}count{suffix}": count,
        f"{prefix}false_negative_rate{suffix}": fn / pos if pos > 0 else float("nan"),
        f"{prefix}false_negative{suffix}": fn,
        f"{prefix}false_positive_rate{suffix}": fp / neg if neg > 0 else float("nan"),
        f"{prefix}false_positive{suffix}": fp,
        f"{prefix}true_negative_rate{suffix}": tn / neg if neg > 0 else float("nan"),
        f"{prefix}true_negative{suffix}": tn,
        f"{prefix}true_positive_rate{suffix}": tp / pos if pos > 0 else float("nan"),
        f"{prefix}true_positive{suffix}": tp,
    }


def multiclass_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the confusion matrix metrics for multiclass labels.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multiclass_confusion_matrix_metrics
    >>> multiclass_confusion_matrix_metrics(
    ...     y_true=np.array([0, 1, 1, 2, 2, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
    ... )
    {'confusion_matrix': array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]), 'count': 6}

    ```
    """
    y_true, y_pred = preprocess_pred(y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=True)
    return {
        f"{prefix}confusion_matrix{suffix}": metrics.confusion_matrix(y_true=y_true, y_pred=y_pred),
        f"{prefix}count{suffix}": y_true.size,
    }


def multilabel_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the confusion matrix metrics for multilabel labels.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import multilabel_confusion_matrix_metrics
    >>> multilabel_confusion_matrix_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ... )
    {'confusion_matrix': array([[[2, 0], [0, 3]],
                                [[3, 0], [0, 2]],
                                [[2, 0], [0, 3]]]),
     'count': 5}

    ```
    """
    y_true, y_pred = preprocess_pred_multilabel(y_true, y_pred, remove_nan=True)
    count = y_true.shape[0]
    if count > 0:
        if y_true.shape[1] > 1:
            confmat = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
        else:
            confmat = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred).reshape(1, 2, 2)
    else:
        confmat = np.zeros((0, 0, 0), dtype=np.int64)
    return {f"{prefix}confusion_matrix{suffix}": confmat, f"{prefix}count{suffix}": count}
