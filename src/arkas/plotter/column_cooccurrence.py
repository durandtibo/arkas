r"""Contain the implementation of a pairwise column co-occurrence
plotter."""

from __future__ import annotations

__all__ = ["ColumnCooccurrencePlotter"]

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from coola import objects_are_equal
from grizz.utils.cooccurrence import compute_pairwise_cooccurrence

from arkas.plotter.base import BasePlotter
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


class ColumnCooccurrencePlotter(BasePlotter):
    r"""Implement a pairwise column co-occurrence plotter.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.plotter import ColumnCooccurrencePlotter
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> plotter = ColumnCooccurrencePlotter(frame)
    >>> plotter
    ColumnCooccurrencePlotter(shape=(7, 3), ignore_self=False)
    """

    def __init__(self, frame: pl.DataFrame, ignore_self: bool = False) -> None:
        self._frame = frame
        self._ignore_self = bool(ignore_self)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(shape={self._frame.shape}, "
            f"ignore_self={self._ignore_self})"
        )

    def compute(self) -> Plotter:
        return Plotter(self.plot())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._ignore_self == other._ignore_self and objects_are_equal(
            self._frame, other._frame, equal_nan=equal_nan
        )

    def plot(self, prefix: str = "", suffix: str = "") -> dict:
        return {f"{prefix}column_cooccurrence{suffix}": plt.subplots()[0]}

    def cooccurrence_matrix(self) -> np.ndarray:
        r"""Return the pairwise column co-occurrence matrix.

        Returns:
            The pairwise column co-occurrence.
        """
        return compute_pairwise_cooccurrence(frame=self._frame, ignore_self=self._ignore_self)
