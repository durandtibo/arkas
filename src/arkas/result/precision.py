r"""Implement the precision result."""

from __future__ import annotations

__all__ = ["PrecisionResult", "precision_metrics"]

from typing import Any

import numpy as np
from coola import objects_are_equal
from sklearn import metrics

from arkas.result.base import BaseResult


class PrecisionResult(BaseResult):
    r"""Implement the precision result.

    This result can be used in 3 different settings:

    - binary: ``y_true`` must be an array of shape ``(n_samples,)``
        with ``0`` and ``1`` values, and ``y_pred`` must be an array
        of shape ``(n_samples,)``.
    - multiclass: ``y_true`` must be an array of shape ``(n_samples,)``
        with values in ``{0, ..., n_classes-1}``, and ``y_pred`` must
        be an array of shape ``(n_samples,)``.
    - multilabel: ``y_true`` must be an array of shape
        ``(n_samples, n_classes)`` with ``0`` and ``1`` values, and
        ``y_pred`` must be an array of shape
        ``(n_samples, n_classes)``.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import PrecisionResult
    >>> # binary
    >>> result = PrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    PrecisionResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'precision': 1.0, 'count': 5}
    >>> # multilabel
    >>> result = PrecisionResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ... )
    >>> result
    PrecisionResult(y_true=(5, 3), y_pred=(5, 3))
    >>> result.compute_metrics()
    {'precision': array([1., 1., 0.]),
     'count': 5,
     'macro_precision': 0.666...,
     'micro_precision': 0.714...,
     'weighted_precision': 0.625}
    >>> # multiclass
    >>> result = PrecisionResult(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ... )
    >>> result
    PrecisionResult(y_true=(6,), y_pred=(6,))
    >>> result.compute_metrics()
    {'precision': array([1., 1., 1.]),
     'count': 6,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'weighted_precision': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true = y_true
        self._y_pred = y_pred

        self._y_true_unique = set(np.unique(self._y_true).tolist())

        self._check_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, y_pred={self._y_pred.shape})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        if self._y_true_unique.issubset({0, 1}):
            if self._y_true.ndim == 1:
                return self._compute_binary_metrics(prefix=prefix, suffix=suffix)
            return self._compute_multilabel_metrics(prefix=prefix, suffix=suffix)
        return self._compute_multiclass_metrics(prefix=prefix, suffix=suffix)

    def _compute_binary_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        count = self._y_true.size
        ap = float("nan")
        if count > 0:
            ap = float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average="binary")
            )
        return {
            f"{prefix}precision{suffix}": ap,
            f"{prefix}count{suffix}": self._y_true.size,
        }

    def _compute_multiclass_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        n_samples = self._y_pred.shape[0] if self._y_pred.ndim > 0 else 0
        n_classes = len(self._y_true_unique)
        ap = np.full((n_classes,), fill_value=float("nan"))
        macro_ap, micro_ap, weighted_ap = [float("nan")] * 3
        if n_samples > 0:
            ap = np.asarray(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average=None)
            ).reshape([n_classes])
            macro_ap = float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average="macro")
            )
            micro_ap = float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average="micro")
            )
            weighted_ap = float(
                metrics.precision_score(
                    y_true=self._y_true, y_pred=self._y_pred, average="weighted"
                )
            )
        return {
            f"{prefix}precision{suffix}": ap,
            f"{prefix}count{suffix}": n_samples,
            f"{prefix}macro_precision{suffix}": macro_ap,
            f"{prefix}micro_precision{suffix}": micro_ap,
            f"{prefix}weighted_precision{suffix}": weighted_ap,
        }

    def _compute_multilabel_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        n_samples, n_classes = self._y_true.shape
        ap = np.full((n_classes,), fill_value=float("nan"))
        macro_ap, micro_ap, weighted_ap = [float("nan")] * 3
        if n_samples > 0:
            ap = np.asarray(
                metrics.precision_score(
                    y_true=self._y_true,
                    y_pred=self._y_pred,
                    average="binary" if n_classes == 1 else None,
                )
            ).reshape([n_classes])
            macro_ap = float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average="macro")
            )
            micro_ap = float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred, average="micro")
            )
            weighted_ap = float(
                metrics.precision_score(
                    y_true=self._y_true, y_pred=self._y_pred, average="weighted"
                )
            )
        return {
            f"{prefix}precision{suffix}": ap,
            f"{prefix}count{suffix}": n_samples,
            f"{prefix}macro_precision{suffix}": macro_ap,
            f"{prefix}micro_precision{suffix}": micro_ap,
            f"{prefix}weighted_precision{suffix}": weighted_ap,
        }

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.y_true, other.y_true, equal_nan=equal_nan
        ) and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.ndim > 2:
            msg = (
                f"'y_true' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_true.shape}"
            )
            raise ValueError(msg)
        if self._y_pred.ndim > 2:
            msg = (
                f"'y_pred' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)
        if self._y_true.ndim == self._y_pred.ndim and self._y_true.shape != self._y_pred.shape:
            msg = (
                f"'y_true' and 'y_pred' have different shapes: {self._y_true.shape} vs "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)


def precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_type: str = "auto",
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
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
    """
    if label_type == "binary":
        return _binary_precision_metrics(
            y_true=y_true.ravel(), y_pred=y_pred.ravel(), prefix=prefix, suffix=suffix
        )
    if label_type == "multiclass":
        return _multiclass_precision_metrics(
            y_true=y_true.ravel(), y_pred=y_pred.ravel(), prefix=prefix, suffix=suffix
        )
    msg = (
        f"Incorrect label type: '{label_type}'. The supported label types are: "
        f"'binary', 'multiclass', 'multilabel', and 'auto'"
    )
    raise RuntimeError(msg)


def _binary_precision_metrics(
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
    """
    if y_true.shape != y_pred.shape:
        msg = f"'y_true' and 'y_pred' have different shapes: {y_true.shape} vs {y_pred.shape}"
        raise RuntimeError(msg)

    count = y_true.size
    precision = float("nan")
    if count > 0:
        precision = float(metrics.precision_score(y_true=y_true, y_pred=y_pred))
    return {f"{prefix}precision{suffix}": precision, f"{prefix}count{suffix}": count}


def _multiclass_precision_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
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
    """
    n_samples = y_true.shape[0]
    macro_precision, micro_precision, weighted_precision = [float("nan")] * 3
    n_classes = y_pred.shape[1] if y_pred.ndim == 2 else 0 if n_samples == 0 else 1
    precision = np.full((n_classes,), fill_value=float("nan"))
    if n_samples > 0:
        macro_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="macro")
        )
        micro_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="micro")
        )
        weighted_precision = float(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average="weighted")
        )
        precision = np.asarray(
            metrics.precision_score(y_true=y_true, y_pred=y_pred, average=None)
        ).ravel()
    return {
        f"{prefix}precision{suffix}": precision,
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_precision{suffix}": macro_precision,
        f"{prefix}micro_precision{suffix}": micro_precision,
        f"{prefix}weighted_precision{suffix}": weighted_precision,
    }
