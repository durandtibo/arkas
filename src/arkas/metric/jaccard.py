r"""Implement the jaccard result."""

from __future__ import annotations

__all__ = ["jaccard_metrics"]


import numpy as np
from sklearn import metrics

from arkas.metric.precision import find_label_type
from arkas.metric.utils import preprocess_true_pred


def jaccard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the jaccard metrics.

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
    >>> from arkas.metric import jaccard_metrics
    >>> # auto
    >>> jaccard_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'count': 5, 'jaccard': 1.0}
    >>> # binary
    >>> jaccard_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     label_type="binary",
    ... )
    {'count': 5, 'jaccard': 1.0}
    >>> # multiclass
    >>> jaccard_metrics(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ...     label_type="multiclass",
    ... )
    {'count': 6,
     'jaccard': array([1., 1., 1.]),
     'macro_jaccard': 1.0,
     'micro_jaccard': 1.0,
     'weighted_jaccard': 1.0}
    >>> # multilabel
    >>> jaccard_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     label_type="multilabel",
    ... )
    {'count': 5,
     'jaccard': array([1., 1., 1.]),
     'macro_jaccard': 1.0,
     'micro_jaccard': 1.0,
     'weighted_jaccard': 1.0}

    ```
    """
    if label_type == "auto":
        label_type = find_label_type(y_true=y_true, y_pred=y_pred)
    if label_type == "binary":
        return _binary_jaccard_metrics(
            y_true=y_true.ravel(), y_pred=y_pred.ravel(), prefix=prefix, suffix=suffix
        )
    if label_type == "multiclass":
        return _multiclass_jaccard_metrics(
            y_true=y_true.ravel(), y_pred=y_pred.ravel(), prefix=prefix, suffix=suffix
        )
    if label_type == "multilabel":
        return _multilabel_jaccard_metrics(
            y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix
        )
    msg = (
        f"Incorrect label type: '{label_type}'. The supported label types are: "
        f"'binary', 'multiclass', 'multilabel', and 'auto'"
    )
    raise RuntimeError(msg)


def _binary_jaccard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the jaccard metrics for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    if y_true.shape != y_pred.shape:
        msg = f"'y_true' and 'y_pred' have different shapes: {y_true.shape} vs {y_pred.shape}"
        raise RuntimeError(msg)

    y_true, y_pred = preprocess_true_pred(y_true=y_true, y_pred=y_pred, nan="remove")

    count, jaccard = y_true.size, float("nan")
    if count > 0:
        jaccard = float(metrics.jaccard_score(y_true=y_true, y_pred=y_pred))
    return {f"{prefix}count{suffix}": count, f"{prefix}jaccard{suffix}": jaccard}


def _multiclass_jaccard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the jaccard metrics for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    y_true, y_pred = preprocess_true_pred(y_true=y_true, y_pred=y_pred, nan="remove")

    n_samples = y_true.shape[0]
    macro_jaccard, micro_jaccard, weighted_jaccard = float("nan"), float("nan"), float("nan")
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else 0 if n_samples == 0 else 1
    jaccard = np.full((n_classes,), fill_value=float("nan"))
    if n_samples > 0:
        macro_jaccard = float(
            metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0.0)
        )
        micro_jaccard = float(
            metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0.0)
        )
        weighted_jaccard = float(
            metrics.jaccard_score(
                y_true=y_true, y_pred=y_pred, average="weighted", zero_division=0.0
            )
        )
        jaccard = np.asarray(
            metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average=None, zero_division=0.0)
        ).ravel()
    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}jaccard{suffix}": jaccard,
        f"{prefix}macro_jaccard{suffix}": macro_jaccard,
        f"{prefix}micro_jaccard{suffix}": micro_jaccard,
        f"{prefix}weighted_jaccard{suffix}": weighted_jaccard,
    }


def _multilabel_jaccard_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float | np.ndarray]:
    r"""Return the jaccard metrics for multilabel labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples, n_classes)``.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.
    """
    n_samples = y_true.shape[0]
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else 0 if n_samples == 0 else 1
    jaccard = np.full((n_classes,), fill_value=float("nan"))
    macro_jaccard, micro_jaccard, weighted_jaccard = [float("nan")] * 3
    if n_samples > 0:
        jaccard = np.array(
            metrics.jaccard_score(
                y_true=y_true,
                y_pred=y_pred,
                average="binary" if n_classes == 1 else None,
            )
        ).ravel()
        macro_jaccard = float(metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average="macro"))
        micro_jaccard = float(metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average="micro"))
        weighted_jaccard = float(
            metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average="weighted")
        )
    return {
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}jaccard{suffix}": jaccard,
        f"{prefix}macro_jaccard{suffix}": macro_jaccard,
        f"{prefix}micro_jaccard{suffix}": micro_jaccard,
        f"{prefix}weighted_jaccard{suffix}": weighted_jaccard,
    }
