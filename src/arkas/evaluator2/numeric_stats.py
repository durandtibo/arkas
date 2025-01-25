r"""Implement an evaluator to compute statistics of numerical column."""

from __future__ import annotations

__all__ = ["NumericStatisticsEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.state.dataframe import DataFrameState
from arkas.utils.stats import compute_statistics_continuous


class NumericStatisticsEvaluator(BaseStateCachedEvaluator[DataFrameState]):
    r"""Implement an evaluator to compute statistics of numerical
    columns.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator2 import NumericStatisticsEvaluator
    >>> from arkas.state import DataFrameState
    >>> dataframe = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     }
    ... )
    >>> evaluator = NumericStatisticsEvaluator(DataFrameState(dataframe))
    >>> evaluator
    NumericStatisticsEvaluator(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> evaluator.evaluate()
    {'col1': {'count': 7, ...}, 'col2': {'count': 7, ...}}

    ```
    """

    def _evaluate(self) -> dict[str, dict[str, float]]:
        return {
            series.name: compute_statistics_continuous(series) for series in self._state.dataframe
        }
