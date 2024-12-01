r"""Implement the Wasserstein distance between two 1D distributions
result."""

from __future__ import annotations

__all__ = ["WassersteinDistanceResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric import wasserstein_distance
from arkas.metric.utils import check_same_shape
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class WassersteinDistanceResult(BaseResult):
    r"""Implement the Wasserstein distance between two 1D distributions
    result.

    Args:
        u_values: The values observed in the (empirical) distribution.
        v_values: The values observed in the (empirical) distribution.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import WassersteinDistanceResult
    >>> result = WassersteinDistanceResult(
    ...     u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ... )
    >>> result
    WassersteinDistanceResult(u_values=(5,), v_values=(5,))
    >>> result.compute_metrics()
    {'count': 5, 'wasserstein_distance': 0.0}

    ```
    """

    def __init__(self, u_values: np.ndarray, v_values: np.ndarray) -> None:
        self._u_values = u_values.ravel()
        self._v_values = v_values.ravel()

        check_same_shape([self._u_values, self._v_values])

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(u_values={self._u_values.shape}, "
            f"v_values={self._v_values.shape})"
        )

    @property
    def u_values(self) -> np.ndarray:
        return self._u_values

    @property
    def v_values(self) -> np.ndarray:
        return self._v_values

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return wasserstein_distance(
            u_values=self._u_values,
            v_values=self._v_values,
            prefix=prefix,
            suffix=suffix,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.u_values, other.u_values, equal_nan=equal_nan
        ) and objects_are_equal(self.v_values, other.v_values, equal_nan=equal_nan)

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}
