r"""Contain the implementation of a DataFrame column plotter."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "MatplotlibFigureCreator", "PlotColumnPlotter"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter.caching import BaseStateCachedPlotter
from arkas.state.dataframe import DataFrameState

if TYPE_CHECKING:
    from arkas.figure.base import BaseFigure


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure with the content of
    each column."""

    @abstractmethod
    def create(self, state: DataFrameState) -> BaseFigure:
        r"""Create a figure with the content of each column.

        Args:
            state: The state containing the DataFrame to analyze.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.state import DataFrameState
        >>> creator = MatplotlibFigureCreator()
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1.2, 4.2, 4.2, 2.2],
        ...         "col2": [1, 1, 1, 1],
        ...         "col3": [1, 2, 2, 2],
        ...     },
        ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        ... )
        >>> fig = creator.create(DataFrameState(frame))

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure with the content of each column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.state import DataFrameState
    >>> creator = MatplotlibFigureCreator()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> fig = creator.create(DataFrameState(frame))

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, state: DataFrameState) -> BaseFigure:
        if state.dataframe.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        fig, ax = plt.subplots(**state.figure_config.get_arg("init", {}))
        for col in state.dataframe:
            ax.plot(col.to_numpy(), label=col.name)

        xmin, xmax = 0, state.dataframe.shape[0] - 1
        if xmin < xmax:
            ax.set_xlim(xmin, xmax)
        if yscale := state.figure_config.get_arg("yscale"):
            ax.set_yscale(yscale)
        ax.legend()
        fig.tight_layout()
        return MatplotlibFigure(fig)


class PlotColumnPlotter(BaseStateCachedPlotter[DataFrameState]):
    r"""Implement a DataFrame column plotter.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.plotter import PlotColumnPlotter
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> plotter = PlotColumnPlotter(DataFrameState(frame))
    >>> plotter
    PlotColumnPlotter(
      (state): DataFrameState(dataframe=(4, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    registry = FigureCreatorRegistry[BaseFigureCreator](
        {MatplotlibFigureConfig.backend(): MatplotlibFigureCreator()}
    )

    def _plot(self) -> dict:
        figure = self.registry.find_creator(self._state.figure_config.backend()).create(self._state)
        return {"plot_column": figure}
