r"""Implement the Pearson correlation result."""

from __future__ import annotations

__all__ = ["PearsonCorrelationResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric.correlation.pearson import pearsonr
from arkas.metric.utils import check_same_shape
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class PearsonCorrelationResult(BaseResult):
    r"""Implement the Pearson correlation result.

    Args:
        x: The first input array.
        y: The second input array.
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
    ...     x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])
    ... )
    >>> result
    PearsonCorrelationResult(x=(5,), y=(5,), alternative=two-sided)
    >>> result.compute_metrics()
    {'count': 5, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0}

    ```
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, alternative: str = "two-sided") -> None:
        self._x = x.ravel()
        self._y = y.ravel()
        self._alternative = alternative

        check_same_shape([self._x, self._y])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(x={self._x.shape}, "
            f"y={self._y.shape}, alternative={self._alternative})"
        )

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def alternative(self) -> str:
        return self._alternative

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return pearsonr(
            x=self._x,
            y=self._y,
            alternative=self._alternative,
            prefix=prefix,
            suffix=suffix,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.x, other.x, equal_nan=equal_nan)
            and objects_are_equal(self.y, other.y, equal_nan=equal_nan)
            and self.alternative == other.alternative
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}
