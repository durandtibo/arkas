r"""Implement an output to analyze a column with continuous values."""

from __future__ import annotations

__all__ = ["TemporalContinuousColumnOutput"]


from arkas.content.continuous_temporal import TemporalContinuousColumnContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.temporal_column import TemporalColumnState


class TemporalContinuousColumnOutput(BaseStateOutput[TemporalColumnState]):
    r"""Implement an output to analyze a column with continuous values.

    Args:
        state: The state containing the column to analyze.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.output import TemporalContinuousColumnOutput
    >>> from arkas.state import TemporalColumnState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 0, 1],
    ...         "col2": [0, 1, 2, 3],
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
    ...         "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...     },
    ... )
    >>> output = TemporalContinuousColumnOutput(
    ...     TemporalColumnState(frame, target_column="col2", temporal_column="datetime")
    ... )
    >>> output
    TemporalContinuousColumnOutput(
      (state): TemporalColumnState(dataframe=(4, 3), target_column='col2', temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    TemporalContinuousColumnContentGenerator(
      (state): TemporalColumnState(dataframe=(4, 3), target_column='col2', temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: TemporalColumnState) -> None:
        super().__init__(state)
        self._content = TemporalContinuousColumnContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> TemporalContinuousColumnContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
