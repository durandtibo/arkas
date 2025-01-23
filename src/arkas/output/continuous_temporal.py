r"""Implement an output to analyze a column with continuous values."""

from __future__ import annotations

__all__ = ["TemporalContinuousColumnOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.content.continuous_temporal import TemporalContinuousColumnContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.lazy import BaseLazyOutput

if TYPE_CHECKING:
    from arkas.state.temporal_dataframe import TemporalDataFrameState


class TemporalContinuousColumnOutput(BaseLazyOutput):
    r"""Implement an output to analyze a column with continuous values.

    Args:
        state: The state containing the column to analyze.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.output import TemporalContinuousColumnOutput
    >>> from arkas.state import TemporalDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 2, 3],
    ...         "datetime": [
    ...             datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...         ],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...     },
    ... )
    >>> output = TemporalContinuousColumnOutput(
    ...     TemporalDataFrameState(frame, temporal_column="datetime")
    ... )
    >>> output
    TemporalContinuousColumnOutput(
      (state): TemporalDataFrameState(dataframe=(4, 2), temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    TemporalContinuousColumnContentGenerator(
      (state): TemporalDataFrameState(dataframe=(4, 2), temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

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

    def _get_content_generator(self) -> TemporalContinuousColumnContentGenerator:
        return TemporalContinuousColumnContentGenerator(self._state)

    def _get_evaluator(self) -> Evaluator:
        return Evaluator()
