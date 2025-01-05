r"""Contain the implementation of a DataFrame column plotter."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "MatplotlibFigureCreator", "PlotColumnPlotter"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter.base import BasePlotter
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure.base import BaseFigure, BaseFigureConfig
    from arkas.state.dataframe import DataFrameState


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure with the content of each
    column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.plot_column import MatplotlibFigureCreator
    >>> creator = MatplotlibFigureCreator()
    >>> creator
    MatplotlibFigureCreator()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> config = MatplotlibFigureConfig()
    >>> fig = creator.create(frame=frame, config=config)

    ```
    """

    @abstractmethod
    def create(self, frame: pl.DataFrame, config: BaseFigureConfig) -> BaseFigure:
        r"""Create a figure with the content of each column.

        Args:
            frame: The input DataFrame.
            config: The figure config.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.plotter.plot_column import MatplotlibFigureCreator
        >>> creator = MatplotlibFigureCreator()
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1.2, 4.2, 4.2, 2.2],
        ...         "col2": [1, 1, 1, 1],
        ...         "col3": [1, 2, 2, 2],
        ...     },
        ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
        ... )
        >>> config = MatplotlibFigureConfig()
        >>> fig = creator.create(frame=frame, config=config)

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure with the content of each column.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.plot_column import MatplotlibFigureCreator
    >>> creator = MatplotlibFigureCreator()
    >>> creator
    MatplotlibFigureCreator()
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> config = MatplotlibFigureConfig()
    >>> fig = creator.create(frame=frame, config=config)

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, frame: pl.DataFrame, config: BaseFigureConfig) -> BaseFigure:
        if frame.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        fig, ax = plt.subplots(**config.get_arg("init", {}))

        for col in frame:
            ax.plot(col.to_numpy(), label=col.name)

        if yscale := config.get_arg("yscale"):
            ax.set_yscale(yscale)

        ax.legend()
        fig.tight_layout()
        return MatplotlibFigure(fig)


class PlotColumnPlotter(BasePlotter):
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
      (state): DataFrameState(dataframe=(4, 3), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    registry = FigureCreatorRegistry[BaseFigureCreator](
        {MatplotlibFigureConfig.backend(): MatplotlibFigureCreator()}
    )

    def __init__(self, state: DataFrameState) -> None:
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def compute(self) -> Plotter:
        return Plotter(self.plot())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)

    def plot(self, prefix: str = "", suffix: str = "") -> dict:
        figure = self.registry.find_creator(self._state.figure_config.backend()).create(
            frame=self._state.dataframe,
            config=self._state.figure_config,
        )
        return {f"{prefix}plot_column{suffix}": figure}
