r"""Implement the accuracy result."""

from __future__ import annotations

__all__ = ["AccuracyResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from sklearn import metrics

from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class AccuracyResult(BaseResult):
    r"""Implement the accuracy result.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` where the values
            are in ``{0, ..., n_classes-1}``.
        y_pred: The predicted labels. This input must be an
            array of shape ``(n_samples,)`` where the values are
            in ``{0, ..., n_classes-1}``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AccuracyResult
    >>> result = AccuracyResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    AccuracyResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'accuracy': 1.0, 'count': 5}

    ```
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()

        self._check_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, y_pred={self._y_pred.shape})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def equal(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.y_true, other.y_true) and objects_are_equal(
            self.y_pred, other.y_pred
        )

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return {
            f"{prefix}accuracy{suffix}": float(
                metrics.accuracy_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
            f"{prefix}count{suffix}": self._y_true.size,
        }

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.shape != self._y_pred.shape:
            msg = (
                f"'y_true' and 'y_pred' have different shapes: {self._y_true.shape} vs "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)
