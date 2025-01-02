r"""Contain the implementation of a pairwise column co-occurrence
plotter."""

from __future__ import annotations

__all__ = ["ColumnCooccurrencePlotter"]

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from coola import objects_are_equal
from grizz.utils.cooccurrence import compute_pairwise_cooccurrence

from arkas.plot.utils import readable_xticklabels, readable_yticklabels
from arkas.plotter.base import BasePlotter
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    def compute(self) -> Plotter:
        return Plotter(self.plot())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._ignore_self == other._ignore_self and objects_are_equal(
            self._frame, other._frame, equal_nan=equal_nan
        )

    def plot(self, prefix: str = "", suffix: str = "") -> dict:
        return {
            f"{prefix}column_cooccurrence{suffix}": create_figure(
                matrix=self.cooccurrence_matrix(),
                columns=self._frame.columns,
            )
        }

    def cooccurrence_matrix(self) -> np.ndarray:
        r"""Return the pairwise column co-occurrence matrix.

        Returns:
            The pairwise column co-occurrence.
        """
        return compute_pairwise_cooccurrence(frame=self._frame, ignore_self=self._ignore_self)


def create_figure(matrix: np.ndarray, columns: Sequence[str]) -> plt.Figure:
    r"""Create a figure of the pairwise column co-occurrence matrix.

    Args:
        matrix: The co-occurrence matrix.
        columns: The column names.

    Returns:
        The generated figure.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.plotter.column_cooccurrence import create_figure
    >>> fig = create_figure(matrix=np.ones((3, 3)), columns=["a", "b", "c"])

    ```
    """
    fig, ax = plt.subplots()
    if matrix.shape[0] == 0:
        return fig

    ax.imshow(matrix)
    ax.set_xticks(
        range(len(columns)),
        labels=columns,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize="x-small" if matrix.shape[0] > 30 else None,
    )
    ax.set_yticks(
        range(len(columns)),
        labels=columns,
        fontsize="x-small" if matrix.shape[0] > 30 else None,
    )
    readable_xticklabels(ax, max_num_xticks=50)
    readable_yticklabels(ax, max_num_yticks=50)
    ax.set_title("pairwise column co-occurrence matrix")

    if matrix.shape[0] < 16:
        for i in range(len(columns)):
            for j in range(len(columns)):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="w", fontsize="x-small")

    fig.tight_layout()
    return fig
