r"""Implement an output to analyze the correlation between columns."""

from __future__ import annotations

__all__ = ["CorrelationOutput"]


from arkas.content.correlation import CorrelationContentGenerator
from arkas.evaluator2.correlation import CorrelationEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.columns import TwoColumnDataFrameState


class CorrelationOutput(BaseStateOutput[TwoColumnDataFrameState]):
    r"""Implement an output to summarize the numeric columns of a
    DataFrame.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import CorrelationOutput
    >>> from arkas.state import TwoColumnDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...     },
    ... )
    >>> output = CorrelationOutput(
    ...     TwoColumnDataFrameState(frame, column1="col1", column2="col2")
    ... )
    >>> output
    CorrelationOutput(
      (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    CorrelationContentGenerator(
      (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    CorrelationEvaluator(
      (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: TwoColumnDataFrameState) -> None:
        super().__init__(state)
        self._content = CorrelationContentGenerator(self._state)
        self._evaluator = CorrelationEvaluator(self._state)

    def _get_content_generator(self) -> CorrelationContentGenerator:
        return self._content

    def _get_evaluator(self) -> CorrelationEvaluator:
        return self._evaluator
