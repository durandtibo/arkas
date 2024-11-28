r"""Implement the Pearson correlation result."""

from __future__ import annotations

__all__ = ["PearsonCorrelationResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric.regression.pearson import pearsonr
from arkas.metric.utils import check_same_shape_pred
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class PearsonCorrelationResult(BaseResult):
    r"""Implement the Pearson correlation result.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        alternative: The alternative hypothesis. Default is 'two-sided'.
            The following options are available:
            - 'two-sided': the correlation is nonzero
            - 'less': the correlation is negative (less than zero)
            - 'greater': the correlation is positive (greater than zero)

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import PearsonCorrelationResult
    >>> result = PearsonCorrelationResult(
    ...     y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
    ... )
    >>> result
    PearsonCorrelationResult(y_true=(5,), y_pred=(5,), alternative=two-sided)
    >>> result.compute_metrics()
    {'count': 5, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0}

    ```
    """

    def __init__(
        self, y_true: np.ndarray, y_pred: np.ndarray, alternative: str = "two-sided"
    ) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()
        self._alternative = alternative

        check_same_shape_pred(y_true=self._y_true, y_pred=self._y_pred)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_pred={self._y_pred.shape}, alternative={self._alternative})"
        )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    @property
    def alternative(self) -> str:
        return self._alternative

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return pearsonr(
            y_true=self._y_true,
            y_pred=self._y_pred,
            alternative=self._alternative,
            prefix=prefix,
            suffix=suffix,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)
            and self.alternative == other.alternative
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}
