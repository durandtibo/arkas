r"""Implement an output to plot each column of a DataFrame along a
temporal dimension."""

from __future__ import annotations

__all__ = ["TemporalPlotColumnOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.content.plot_column import PlotColumnContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.lazy import BaseLazyOutput
from arkas.plotter.plot_column import PlotColumnPlotter

if TYPE_CHECKING:
    from arkas.state.temporal_dataframe import TemporalDataFrameState


class TemporalPlotColumnOutput(BaseLazyOutput):
    r"""Implement an output to plot each column of a DataFrame along a
    temporal dimension.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.output import TemporalPlotColumnOutput
    >>> from arkas.state import TemporalDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0],
    ...         "col2": [0, 1, 0, 1],
    ...         "col3": [1, 0, 0, 0],
    ...         "datetime": [
    ...             datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...         ],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "col2": pl.Int64,
    ...         "col3": pl.Int64,
    ...         "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...     },
    ... )
    >>> output = TemporalPlotColumnOutput(
    ...     TemporalDataFrameState(frame, temporal_column="datetime")
    ... )
    >>> output
    TemporalPlotColumnOutput(
      (state): TemporalDataFrameState(dataframe=(4, 4), temporal_column='datetime', period=None, figure_config=MatplotlibFigureConfig(color_norm=None))
    )
    >>> output.get_content_generator()
    PlotColumnContentGenerator(
      (state): TemporalDataFrameState(dataframe=(4, 4), temporal_column='datetime', period=None, figure_config=MatplotlibFigureConfig(color_norm=None))
    )
    >>> output.get_evaluator()
    Evaluator(count=0)
    >>> output.get_plotter()
    PlotColumnPlotter(
      (state): TemporalDataFrameState(dataframe=(4, 4), temporal_column='datetime', period=None, figure_config=MatplotlibFigureConfig(color_norm=None))
    )

    ```
    """

    def __init__(self, state: TemporalDataFrameState) -> None:
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
        return self._state.equal(other._state, equal_nan=equal_nan)

    def _get_content_generator(self) -> PlotColumnContentGenerator:
        return PlotColumnContentGenerator(self._state)

    def _get_evaluator(self) -> Evaluator:
        return Evaluator()

    def _get_plotter(self) -> PlotColumnPlotter:
        return PlotColumnPlotter(self._state)
