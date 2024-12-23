r"""Implement the Jensen-Shannon (JS) divergence between two 1D
distributions result."""

from __future__ import annotations

__all__ = ["JensenShannonDivergenceResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric import jensen_shannon_divergence
from arkas.metric.utils import check_same_shape
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import numpy as np


class JensenShannonDivergenceResult(BaseResult):
    r"""Implement the Jensen-Shannon (JS) divergence between two 1D
    distributions result.

    Args:
        p: The true probability distribution.
        q: The model probability distribution.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import JensenShannonDivergenceResult
    >>> result = JensenShannonDivergenceResult(
    ...     p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])
    ... )
    >>> result
    JensenShannonDivergenceResult(p=(4,), q=(4,))
    >>> result.compute_metrics()
    {'size': 4, 'jensen_shannon_divergence': 0.027...}

    ```
    """

    def __init__(self, p: np.ndarray, q: np.ndarray) -> None:
        self._p = p.ravel()
        self._q = q.ravel()

        check_same_shape([self._p, self._q])

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(p={self._p.shape}, q={self._q.shape})"

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def q(self) -> np.ndarray:
        return self._q

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return jensen_shannon_divergence(p=self._p, q=self._q, prefix=prefix, suffix=suffix)

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.p, other.p, equal_nan=equal_nan) and objects_are_equal(
            self.q, other.q, equal_nan=equal_nan
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}