r"""Contain the implementation of a correlation plotter."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "CorrelationPlotter", "MatplotlibFigureCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plot.utils.scatter import find_alpha_from_size, find_marker_size_from_size
from arkas.plotter.caching import BaseStateCachedPlotter
from arkas.state.columns import TwoColumnDataFrameState
from arkas.utils.range import find_range

if TYPE_CHECKING:
    from arkas.figure.base import BaseFigure


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure with the content of
    each column."""

    @abstractmethod
    def create(self, state: TwoColumnDataFrameState) -> BaseFigure:
        r"""Create a figure with the content of each column.

        Args:
            state: The state containing the DataFrame to analyze.
                The DataFrame must have only 2 columns, which are the
                two columns to analyze.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.state import TwoColumnDataFrameState
        >>> from arkas.plotter.correlation import MatplotlibFigureCreator
        >>> creator = MatplotlibFigureCreator()
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ...         "col2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ...     },
        ... )
        >>> fig = creator.create(TwoColumnDataFrameState(frame, column1="col1", column2="col2"))

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure with the content of each column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.state import TwoColumnDataFrameState
    >>> from arkas.plotter.correlation import MatplotlibFigureCreator
    >>> creator = MatplotlibFigureCreator()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     },
    ... )
    >>> fig = creator.create(TwoColumnDataFrameState(frame, column1="col1", column2="col2"))

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, state: TwoColumnDataFrameState) -> BaseFigure:
        if state.dataframe.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        xcol, ycol = state.column1, state.column2

        fig, ax = plt.subplots(**state.figure_config.get_arg("init", {}))
        x = state.dataframe[xcol].to_numpy()
        y = state.dataframe[ycol].to_numpy()

        n = x.size
        marker_alpha = state.get_arg("marker_alpha", default=find_alpha_from_size(n))
        marker_scale = state.get_arg("marker_scale", default=find_marker_size_from_size(n))
        ax.scatter(x=x, y=y, s=marker_scale, alpha=marker_alpha)

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
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        if xscale := state.figure_config.get_arg("xscale"):
            ax.set_xscale(xscale)
        if yscale := state.figure_config.get_arg("yscale"):
            ax.set_yscale(yscale)
        fig.tight_layout()
        return MatplotlibFigure(fig)


class CorrelationPlotter(BaseStateCachedPlotter[TwoColumnDataFrameState]):
    r"""Implement a DataFrame column plotter.

    Args:
        state: The state containing the DataFrame to analyze.
            The DataFrame must have only 2 columns, which are the two
            columns to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.plotter import CorrelationPlotter
    >>> from arkas.state import TwoColumnDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     },
    ... )
    >>> plotter = CorrelationPlotter(
    ...     TwoColumnDataFrameState(frame, column1="col1", column2="col2")
    ... )
    >>> plotter
    CorrelationPlotter(
      (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    registry = FigureCreatorRegistry[BaseFigureCreator](
        {MatplotlibFigureConfig.backend(): MatplotlibFigureCreator()}
    )

    def __init__(self, state: TwoColumnDataFrameState) -> None:
        super().__init__(state)

    def _plot(self) -> dict:
        figure = self.registry.find_creator(self._state.figure_config.backend()).create(self._state)
        return {"correlation": figure}
