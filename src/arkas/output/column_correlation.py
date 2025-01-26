r"""Implement an output to analyze the correlation between columns."""

from __future__ import annotations

__all__ = ["ColumnCorrelationOutput"]


from arkas.content.column_correlation import ColumnCorrelationContentGenerator
from arkas.evaluator2.column_correlation import ColumnCorrelationEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.target_dataframe import TargetDataFrameState


class ColumnCorrelationOutput(BaseStateOutput[TargetDataFrameState]):
    r"""Implement an output to summarize the numeric columns of a
    DataFrame.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import ColumnCorrelationOutput
    >>> from arkas.state import TargetDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ... )
    >>> output = ColumnCorrelationOutput(TargetDataFrameState(frame, target_column="col3"))
    >>> output
    ColumnCorrelationOutput(
      (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    ColumnCorrelationContentGenerator(
      (evaluator): ColumnCorrelationEvaluator(
          (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
        )
    )
    >>> output.get_evaluator()
    ColumnCorrelationEvaluator(
      (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: TargetDataFrameState) -> None:
        super().__init__(state)
        self._evaluator = ColumnCorrelationEvaluator(self._state)
        self._content = ColumnCorrelationContentGenerator(self._evaluator)

    def _get_content_generator(self) -> ColumnCorrelationContentGenerator:
        return self._content

    def _get_evaluator(self) -> ColumnCorrelationEvaluator:
        return self._evaluator
