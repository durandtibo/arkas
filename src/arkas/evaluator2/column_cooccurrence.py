r"""Implement the pairwise column co-occurrence evaluator."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceEvaluator"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from grizz.utils.cooccurrence import compute_pairwise_cooccurrence

from arkas.evaluator2.base import BaseEvaluator
from arkas.evaluator2.vanilla import Evaluator

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


class ColumnCooccurrenceEvaluator(BaseEvaluator):
    r"""Implement the pairwise column co-occurrence evaluator.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator2 import ColumnCooccurrenceEvaluator
    >>> dataframe = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> evaluator = ColumnCooccurrenceEvaluator(dataframe)
    >>> evaluator
    ColumnCooccurrenceEvaluator(shape=(7, 3), ignore_self=False)
    >>> evaluator.evaluate()
    {'column_cooccurrence': array([[3, 2, 1],
           [2, 3, 1],
           [1, 1, 3]])}

    ```
    """

    def __init__(self, frame: pl.DataFrame, ignore_self: bool = False) -> None:
        self._frame = frame
        self._ignore_self = bool(ignore_self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(shape={self._frame.shape}, "
            f"ignore_self={self._ignore_self})"
        )

    @property
    def frame(self) -> pl.DataFrame:
        r"""The DataFrame to analyze."""
        return self._frame

    @property
    def ignore_self(self) -> bool:
        return self._ignore_self

    def compute(self) -> Evaluator:
        return Evaluator(metrics=self.evaluate())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.ignore_self == other.ignore_self and objects_are_equal(
            self.frame, other.frame, equal_nan=equal_nan
        )

    def evaluate(self, prefix: str = "", suffix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}column_cooccurrence{suffix}": compute_pairwise_cooccurrence(
                frame=self._frame, ignore_self=self._ignore_self
            )
        }
