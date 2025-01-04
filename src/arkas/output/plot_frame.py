r"""Implement an output to plot each column of a DataFrame."""

from __future__ import annotations

__all__ = ["PlotDataFrameOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.content import ContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.lazy import BaseLazyOutput
from arkas.plotter.plot_frame import PlotDataFramePlotter

if TYPE_CHECKING:
    from arkas.state.dataframe import DataFrameState


class PlotDataFrameOutput(BaseLazyOutput):
    r"""Implement an output to plot each column of a DataFrame.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import PlotDataFrameOutput
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> output = PlotDataFrameOutput(DataFrameState(frame))
    >>> output
    PlotDataFrameOutput(
      (state): DataFrameState(dataframe=(4, 3), figure_config=MatplotlibFigureConfig(color_norm=None))
    )
    >>> output.get_content_generator()
    ContentGenerator()
    >>> output.get_evaluator()
    Evaluator(count=0)
    >>> output.get_plotter()
    PlotDataFramePlotter(
      (state): DataFrameState(dataframe=(4, 3), figure_config=MatplotlibFigureConfig(color_norm=None))
    )

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(self._state, equal_nan=equal_nan)

    def _get_content_generator(self) -> ContentGenerator:
        return ContentGenerator()

    def _get_evaluator(self) -> Evaluator:
        return Evaluator()

    def _get_plotter(self) -> PlotDataFramePlotter:
        return PlotDataFramePlotter(self._state)
