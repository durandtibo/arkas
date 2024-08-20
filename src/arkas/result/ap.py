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
            be an array of shape ``(n_samples,)`` where the values
            are in ``{0, ..., n_classes-1}``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AveragePrecisionResult
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5,), y_score=(5,))
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self._y_true = y_true.ravel()
        self._y_score = y_score.ravel().astype(np.float64)

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
        count = self._y_true.size
        ap = float("nan")
        if count > 0:
            ap = float(metrics.average_precision_score(y_true=self._y_true, y_score=self._y_score))
        return {
            f"{prefix}average_precision{suffix}": ap,
            f"{prefix}count{suffix}": self._y_true.size,
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
        if self._y_true.shape != self._y_score.shape:
            msg = (
                f"'y_true' and 'y_score' have different shapes: {self._y_true.shape} vs "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
