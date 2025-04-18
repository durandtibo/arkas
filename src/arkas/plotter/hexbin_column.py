r"""Contain the implementation of a DataFrame column plotter that makes
a 2D hexagonal binning plot of points x, y.."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "HexbinColumnPlotter", "MatplotlibFigureCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter.caching import BaseStateCachedPlotter
from arkas.state.scatter_dataframe import ScatterDataFrameState
from arkas.utils.range import find_range

if TYPE_CHECKING:
    from arkas.figure.base import BaseFigure


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure with the content of
    each column."""

    @abstractmethod
    def create(self, state: ScatterDataFrameState) -> BaseFigure:
        r"""Create a figure with the content of each column.

        Args:
            state: The state containing the DataFrame to analyze.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.state import ScatterDataFrameState
        >>> creator = MatplotlibFigureCreator()
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1.2, 4.2, 4.2, 2.2],
        ...         "col2": [1, 1, 1, 1],
        ...         "col3": [1, 2, 2, 2],
        ...     },
        ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        ... )
        >>> fig = creator.create(ScatterDataFrameState(frame, x="col1", y="col2", color="col3"))

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure with the content of each column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.state import ScatterDataFrameState
    >>> creator = MatplotlibFigureCreator()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> fig = creator.create(ScatterDataFrameState(frame, x="col1", y="col2", color="col3"))

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, state: ScatterDataFrameState) -> BaseFigure:
        if state.dataframe.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        fig, ax = plt.subplots(**state.figure_config.get_arg("init", {}))
        color = state.dataframe[state.color].to_numpy() if state.color else None
        x = state.dataframe[state.x].to_numpy()
        y = state.dataframe[state.y].to_numpy()
        s = ax.hexbin(x=x, y=y, C=color)
        fig.colorbar(s)

        xmin, xmax = find_range(
            x,
            xmin=state.figure_config.get_arg("xmin"),
            xmax=state.figure_config.get_arg("xmax"),
        )
        if xmin < xmax:
            ax.set_xlim(xmin, xmax)
        ymin, ymax = find_range(
            y,
            xmin=state.figure_config.get_arg("ymin"),
            xmax=state.figure_config.get_arg("ymax"),
        )
        if ymin < ymax:
            ax.set_ylim(ymin, ymax)
        ax.set_xlabel(state.x)
        ax.set_ylabel(state.y)
        if xscale := state.figure_config.get_arg("xscale"):
            ax.set_xscale(xscale)
        if yscale := state.figure_config.get_arg("yscale"):
            ax.set_yscale(yscale)
        fig.tight_layout()
        return MatplotlibFigure(fig)


class HexbinColumnPlotter(BaseStateCachedPlotter[ScatterDataFrameState]):
    r"""Implement a DataFrame column plotter that makes a 2D hexagonal
    binning plot of points x, y.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.plotter import HexbinColumnPlotter
    >>> from arkas.state import ScatterDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> plotter = HexbinColumnPlotter(
    ...     ScatterDataFrameState(frame, x="col1", y="col2", color="col3")
    ... )
    >>> plotter
    HexbinColumnPlotter(
      (state): ScatterDataFrameState(dataframe=(4, 3), x='col1', y='col2', color='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    registry = FigureCreatorRegistry[BaseFigureCreator](
        {MatplotlibFigureConfig.backend(): MatplotlibFigureCreator()}
    )

    def _plot(self) -> dict:
        figure = self.registry.find_creator(self._state.figure_config.backend()).create(self._state)
        return {"hexbin_column": figure}
