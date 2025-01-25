r"""Implement an analyzer that  analyzes a column with continuous
values."""

from __future__ import annotations

__all__ = ["TemporalContinuousColumnAnalyzer"]

import logging
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.analyzer.lazy import BaseLazyAnalyzer
from arkas.metric.utils import check_nan_policy
from arkas.output.continuous_temporal import TemporalContinuousColumnOutput
from arkas.state.temporal_column import TemporalColumnState

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure import BaseFigureConfig

logger = logging.getLogger(__name__)


class TemporalContinuousColumnAnalyzer(BaseLazyAnalyzer):
    r"""Implement an analyzer that analyzes the temporal distribution of
    a column with continuous values.

    Args:
        target_column: The column to analyze.
        temporal_column: The temporal column in the DataFrame.
        period: An optional temporal period e.g. monthly or daily.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.
        figure_config: The figure configuration.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.analyzer import TemporalContinuousColumnAnalyzer
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
    >>> analyzer = TemporalContinuousColumnAnalyzer(
    ...     target_column="col2", temporal_column="datetime"
    ... )
    >>> analyzer
    TemporalContinuousColumnAnalyzer(target_column='col2', temporal_column='datetime', period=None, nan_policy='propagate', figure_config=None)
    >>> output = analyzer.analyze(frame)
    >>> output
    TemporalContinuousColumnOutput(
      (state): TemporalColumnState(dataframe=(4, 3), target_column='col2', temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(
        self,
        target_column: str,
        temporal_column: str,
        period: str | None = None,
        nan_policy: str = "propagate",
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        self._target_column = target_column
        self._temporal_column = temporal_column
        self._period = period
        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy
        self._figure_config = figure_config

    def __repr__(self) -> str:
        args = repr_mapping_line(self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def get_args(self) -> dict:
        return {
            "target_column": self._target_column,
            "temporal_column": self._temporal_column,
            "period": self._period,
            "nan_policy": self._nan_policy,
            "figure_config": self._figure_config,
        }

    def _analyze(self, frame: pl.DataFrame) -> TemporalContinuousColumnOutput:
        logger.info(
            f"Analyzing the temporal distribution of continuous column {self._target_column!r}..."
        )
        return TemporalContinuousColumnOutput(
            state=TemporalColumnState(
                dataframe=frame,
                target_column=self._target_column,
                temporal_column=self._temporal_column,
                period=self._period,
                nan_policy=self._nan_policy,
                figure_config=self._figure_config,
            )
        )
