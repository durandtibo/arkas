r"""Implement the R^2 (coefficient of determination) regression score
result."""

from __future__ import annotations

__all__ = ["R2ScoreResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric.regression.r2 import r2_score
from arkas.metric.utils import check_same_shape_pred
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class R2ScoreResult(BaseResult):
    r"""Implement the R^2 (coefficient of determination) regression score
    result.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import R2ScoreResult
    >>> result = R2ScoreResult(
    ...     y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
    ... )
    >>> result
    R2ScoreResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'count': 5, 'r2_score': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()

        check_same_shape_pred(y_true=self._y_true, y_pred=self._y_pred)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_pred={self._y_pred.shape})"
        )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return r2_score(y_true=self._y_true, y_pred=self._y_pred, prefix=prefix, suffix=suffix)

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
