r"""Implement the DataFrame summary output."""

from __future__ import annotations

__all__ = ["SummaryOutput"]


from arkas.content.summary import SummaryContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.dataframe import DataFrameState


class SummaryOutput(BaseStateOutput[DataFrameState]):
    r"""Implement the DataFrame summary output.

    Args:
        frame: The DataFrame to analyze.
        top: The number of most frequent values to show.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import SummaryOutput
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     }
    ... )
    >>> output = SummaryOutput(DataFrameState(frame))
    >>> output
    SummaryOutput(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    SummaryContentGenerator(
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        super().__init__(state)
        self._content = SummaryContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> SummaryContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
