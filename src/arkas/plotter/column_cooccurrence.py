r"""Contain the implementation of a pairwise column co-occurrence
plotter."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "ColumnCooccurrencePlotter", "MatplotlibFigureCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from coola import objects_are_equal
from grizz.utils.cooccurrence import compute_pairwise_cooccurrence

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.default import DefaultFigureConfig
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plot.utils import readable_xticklabels, readable_yticklabels
from arkas.plotter.base import BasePlotter
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    import polars as pl

    from arkas.figure.base import BaseFigure, BaseFigureConfig


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure of the pairwise column
    co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
    >>> creator = MatplotlibFigureCreator()
    >>> creator
    MatplotlibFigureCreator()
    >>> config = MatplotlibFigureConfig()
    >>> fig = creator.create(matrix=np.ones((3, 3)), columns=["a", "b", "c"], config=config)

    ```
    """

    @abstractmethod
    def create(
        self, matrix: np.ndarray, columns: Sequence[str], config: BaseFigureConfig
    ) -> BaseFigure:
        r"""Create a figure of the pairwise column co-occurrence matrix.

        Args:
            matrix: The co-occurrence matrix.
            columns: The column names.
            config: The figure config.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
        >>> creator = MatplotlibFigureCreator()
        >>> config = MatplotlibFigureConfig()
        >>> fig = creator.create(matrix=np.ones((3, 3)), columns=["a", "b", "c"], config=config)

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure of the pairwise column co-occurrence
    matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
    >>> creator = MatplotlibFigureCreator()
    >>> creator
    MatplotlibFigureCreator()
    >>> config = MatplotlibFigureConfig()
    >>> fig = creator.create(matrix=np.ones((3, 3)), columns=["a", "b", "c"], config=config)

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(
        self, matrix: np.ndarray, columns: Sequence[str], config: BaseFigureConfig
    ) -> BaseFigure:
        if matrix.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        fontsize = (
            "xx-small" if matrix.shape[0] > 30 else "x-small" if matrix.shape[0] > 15 else "small"
        )
        fig, ax = plt.subplots(**config.get_args())
        im = ax.imshow(matrix)
        fig.colorbar(im, anchor=(-0.3, 0.0), aspect=30).ax.tick_params(labelsize=6)
        ax.set_xticks(
            range(len(columns)),
            labels=columns,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        ax.set_yticks(
            range(len(columns)),
            labels=columns,
            fontsize=fontsize,
        )
        readable_xticklabels(ax, max_num_xticks=50)
        readable_yticklabels(ax, max_num_yticks=50)
        ax.set_title(
            "pairwise column co-occurrence matrix",
            fontsize=None if fontsize == "small" else "small",
        )

        if matrix.shape[0] < 16:
            for i in range(len(columns)):
                for j in range(len(columns)):
                    ax.text(
                        j, i, matrix[i, j], ha="center", va="center", color="w", fontsize="xx-small"
                    )

        fig.tight_layout()
        return MatplotlibFigure(fig)


class ColumnCooccurrencePlotter(BasePlotter):
    r"""Implement a pairwise column co-occurrence plotter.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.
        figure_config: The figure configuration.

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

    registry: FigureCreatorRegistry = FigureCreatorRegistry(
        {
            DefaultFigureConfig.backend(): MatplotlibFigureCreator(),
            MatplotlibFigureConfig.backend(): MatplotlibFigureCreator(),
        }
    )

    def __init__(
        self,
        frame: pl.DataFrame,
        ignore_self: bool = False,
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        self._frame = frame
        self._ignore_self = bool(ignore_self)
        self._figure_config = figure_config or DefaultFigureConfig()

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
        return (
            self._ignore_self == other._ignore_self
            and objects_are_equal(self._frame, other._frame, equal_nan=equal_nan)
            and self._figure_config.equal(other._figure_config)
        )

    def plot(self, prefix: str = "", suffix: str = "") -> dict:
        figure = self.registry.find_creator(self._figure_config.backend()).create(
            matrix=self.cooccurrence_matrix(),
            columns=self._frame.columns,
            config=self._figure_config,
        )
        return {f"{prefix}column_cooccurrence{suffix}": figure}

    def cooccurrence_matrix(self) -> np.ndarray:
        r"""Return the pairwise column co-occurrence matrix.

        Returns:
            The pairwise column co-occurrence.
        """
        return compute_pairwise_cooccurrence(frame=self._frame, ignore_self=self._ignore_self)
