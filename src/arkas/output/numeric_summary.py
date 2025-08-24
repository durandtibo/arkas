r"""Implement an output to summarize the numeric columns of a
DataFrame."""

from __future__ import annotations

__all__ = ["NumericSummaryOutput"]


from arkas.content.numeric_summary import NumericSummaryContentGenerator
from arkas.evaluator2.numeric_stats import NumericStatisticsEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.dataframe import DataFrameState


class NumericSummaryOutput(BaseStateOutput[DataFrameState]):
    r"""Implement an output to summarize the numeric columns of a
    DataFrame.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import NumericSummaryOutput
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ...     schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    ... )
    >>> output = NumericSummaryOutput(DataFrameState(frame))
    >>> output
    NumericSummaryOutput(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    NumericSummaryContentGenerator(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    NumericStatisticsEvaluator(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        super().__init__(state)
        self._content = NumericSummaryContentGenerator(self._state)
        self._evaluator = NumericStatisticsEvaluator(self._state)

    def _get_content_generator(self) -> NumericSummaryContentGenerator:
        return self._content

    def _get_evaluator(self) -> NumericStatisticsEvaluator:
        return self._evaluator
