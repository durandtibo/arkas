r"""Implement the average precision result."""

from __future__ import annotations

__all__ = ["AveragePrecisionResult", "average_precision_metrics"]

from typing import Any

import numpy as np
from coola import objects_are_equal
from sklearn import metrics

from arkas.result.base import BaseResult


class AveragePrecisionResult(BaseResult):
    r"""Implement the average precision result.

    This result can be used in 3 different settings:

    - binary: ``y_true`` must be an array of shape ``(*)``
        with ``0`` and ``1`` values, and ``y_score`` must be an array
        of shape ``(*)``.
    - multiclass: ``y_true`` must be an array of shape ``(n_samples,)``
        with values in ``{0, ..., n_classes-1}``, and ``y_score`` must
        be an array of shape ``(n_samples, n_classes)``.
    - multilabel: ``y_true`` must be an array of shape
        ``(n_samples, n_classes)`` with ``0`` and ``1`` values, and
        ``y_score`` must be an array of shape
        ``(n_samples, n_classes)``.

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

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AveragePrecisionResult
    >>> # binary
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_score=np.array([2, -1, 0, 3, 1]),
    ...     label_type="binary",
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5,), y_score=(5,), label_type=binary)
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}
    >>> # multilabel
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
    ...     label_type="multilabel",
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5, 3), y_score=(5, 3), label_type=multilabel)
    >>> result.compute_metrics()
    {'average_precision': array([1. , 1. , 0.477...]),
     'count': 5,
     'macro_average_precision': 0.825...,
     'micro_average_precision': 0.588...,
     'weighted_average_precision': 0.804...}
    >>> # multiclass
    >>> result = AveragePrecisionResult(
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
    >>> result
    AveragePrecisionResult(y_true=(6,), y_score=(6, 3), label_type=multiclass)
    >>> result.compute_metrics()
    {'average_precision': array([0.833..., 0.75 , 0.75 ]),
     'count': 6,
     'macro_average_precision': 0.777...,
     'micro_average_precision': 0.75,
     'weighted_average_precision': 0.777...}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray, label_type: str) -> None:
        self._y_true = y_true
        self._y_score = y_score.astype(np.float64)
        self._label_type = label_type

        self._check_inputs()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_score={self._y_score.shape}, label_type={self._label_type})"
        )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_score(self) -> np.ndarray:
        return self._y_score

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        if self._y_true.ndim == 1 and self._y_score.ndim == 1:
            return self._compute_binary_metrics(prefix=prefix, suffix=suffix)
        if self._y_true.ndim == 1:
            return self._compute_multiclass_metrics(prefix=prefix, suffix=suffix)
        return self._compute_multilabel_metrics(prefix=prefix, suffix=suffix)

    def _compute_binary_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        count = self._y_true.size
        ap = float("nan")
        if count > 0:
            ap = float(metrics.average_precision_score(y_true=self._y_true, y_score=self._y_score))
        return {
            f"{prefix}average_precision{suffix}": ap,
            f"{prefix}count{suffix}": self._y_true.size,
        }

    def _compute_multiclass_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        n_samples, n_classes = self._y_score.shape
        ap = np.full((n_classes,), fill_value=float("nan"))
        macro_ap, micro_ap, weighted_ap = [float("nan")] * 3
        if n_samples > 0:
            ap = np.asarray(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average=None
                )
            ).reshape([n_classes])
            macro_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="macro"
                )
            )
            micro_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="micro"
                )
            )
            weighted_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="weighted"
                )
            )
        return {
            f"{prefix}average_precision{suffix}": ap,
            f"{prefix}count{suffix}": n_samples,
            f"{prefix}macro_average_precision{suffix}": macro_ap,
            f"{prefix}micro_average_precision{suffix}": micro_ap,
            f"{prefix}weighted_average_precision{suffix}": weighted_ap,
        }

    def _compute_multilabel_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        n_samples, n_classes = self._y_true.shape
        ap = np.full((n_classes,), fill_value=float("nan"))
        macro_ap, micro_ap, weighted_ap = [float("nan")] * 3
        if n_samples > 0:
            ap = np.asarray(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average=None
                )
            ).reshape([n_classes])
            macro_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="macro"
                )
            )
            micro_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="micro"
                )
            )
            weighted_ap = float(
                metrics.average_precision_score(
                    y_true=self._y_true, y_score=self._y_score, average="weighted"
                )
            )
        return {
            f"{prefix}average_precision{suffix}": ap,
            f"{prefix}count{suffix}": n_samples,
            f"{prefix}macro_average_precision{suffix}": macro_ap,
            f"{prefix}micro_average_precision{suffix}": micro_ap,
            f"{prefix}weighted_average_precision{suffix}": weighted_ap,
        }

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_score, other.y_score, equal_nan=equal_nan)
            and self._label_type == other._label_type
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.ndim not in {1, 2}:
            msg = (
                f"'y_true' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_true.shape}"
            )
            raise ValueError(msg)
        if self._y_score.ndim not in {1, 2}:
            msg = (
                f"'y_score' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
        if self._y_true.ndim == self._y_score.ndim and self._y_true.shape != self._y_score.shape:
            msg = (
                f"'y_true' and 'y_score' have different shapes: {self._y_true.shape} vs "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
        if self._label_type not in {"binary", "multiclass", "multilabel"}:
            msg = (
                f"Incorrect label type: '{self._label_type}'. The supported label types are: "
                f"'binary', 'multiclass', 'multilabel'"
            )
            raise ValueError(msg)


def average_precision_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_type: str,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the average precision metrics.

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
    >>> from arkas.result.ap import average_precision_metrics
    >>> # binary
    >>> metrics = average_precision_metrics(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_score=np.array([2, -1, 0, 3, 1]),
    ...     label_type="binary",
    ... )
    >>> metrics
    {'average_precision': 1.0, 'count': 5}
    >>> # multiclass
    >>> metrics = average_precision_metrics(
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
    {'average_precision': array([0.833..., 0.75 , 0.75 ]),
     'count': 6,
     'macro_average_precision': 0.777...,
     'micro_average_precision': 0.75,
     'weighted_average_precision': 0.777...}
    >>> # multilabel
    >>> metrics = average_precision_metrics(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
    ...     label_type="multilabel",
    ... )
    >>> metrics
    {'average_precision': array([1. , 1. , 0.477...]),
     'count': 5,
     'macro_average_precision': 0.825...,
     'micro_average_precision': 0.588...,
     'weighted_average_precision': 0.804...}

    ```
    """
    if label_type == "binary":
        return _binary_average_precision_metrics(
            y_true=y_true.ravel(), y_score=y_score.ravel(), prefix=prefix, suffix=suffix
        )
    if label_type in {"multiclass", "multilabel"}:
        return _multi_average_precision_metrics(
            y_true=y_true, y_score=y_score, prefix=prefix, suffix=suffix
        )
    msg = (
        f"Incorrect label type: '{label_type}'. The supported label types are: "
        f"'binary', 'multiclass', 'multilabel'"
    )
    raise RuntimeError(msg)


def _binary_average_precision_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the average precision metrics for binary labels.

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
    if y_true.shape != y_score.shape:
        msg = f"'y_true' and 'y_score' have different shapes: {y_true.shape} vs {y_score.shape}"
        raise RuntimeError(msg)

    count = y_true.size
    ap = float("nan")
    if count > 0:
        ap = float(metrics.average_precision_score(y_true=y_true, y_score=y_score))
    return {f"{prefix}average_precision{suffix}": ap, f"{prefix}count{suffix}": count}


def _multi_average_precision_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the average precision metrics for multilabel or multiclass
    labels.

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
    n_samples = y_true.shape[0]
    macro_ap, micro_ap, weighted_ap = [float("nan")] * 3
    ap = np.full((y_true.shape[1] if y_true.ndim == 2 else 1,), fill_value=float("nan"))
    if n_samples > 0:
        macro_ap = float(
            metrics.average_precision_score(y_true=y_true, y_score=y_score, average="macro")
        )
        micro_ap = float(
            metrics.average_precision_score(y_true=y_true, y_score=y_score, average="micro")
        )
        weighted_ap = float(
            metrics.average_precision_score(y_true=y_true, y_score=y_score, average="weighted")
        )
        ap = np.asarray(
            metrics.average_precision_score(y_true=y_true, y_score=y_score, average=None)
        ).ravel()

    return {
        f"{prefix}average_precision{suffix}": ap,
        f"{prefix}count{suffix}": n_samples,
        f"{prefix}macro_average_precision{suffix}": macro_ap,
        f"{prefix}micro_average_precision{suffix}": micro_ap,
        f"{prefix}weighted_average_precision{suffix}": weighted_ap,
    }
