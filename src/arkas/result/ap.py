r"""Implement the average precision result."""

from __future__ import annotations

__all__ = ["AveragePrecisionResult"]

from typing import Any

import numpy as np
from coola import objects_are_equal
from sklearn import metrics

from arkas.result.base import BaseResult


class AveragePrecisionResult(BaseResult):
    r"""Implement the average precision result.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            `(n_samples, n_classes)`` where the values
            are ``0`` or ``1``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This array must
            have the same shape as ``y_true``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AveragePrecisionResult
    >>> # binary
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5,), y_score=(5,))
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}
    >>> # multilabel
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5, 3), y_score=(5, 3))
    >>> result.compute_metrics()
    {'average_precision': array([1. , 1. , 0.477...]),
     'count': 5,
     'macro_average_precision': 0.825...,
     'micro_average_precision': 0.588...,
     'weighted_average_precision': 0.804...}


    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self._y_true = y_true
        self._y_score = y_score.astype(np.float64)

        self._check_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, y_score={self._y_score.shape})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_score(self) -> np.ndarray:
        return self._y_score

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        if self._y_true.ndim == 1:
            return self._compute_binary_metrics(prefix=prefix, suffix=suffix)
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

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.y_true, other.y_true) and objects_are_equal(
            self.y_score, other.y_score
        )

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
        if self._y_true.shape != self._y_score.shape:
            msg = (
                f"'y_true' and 'y_score' have different shapes: {self._y_true.shape} vs "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
