r"""Implement an output to plot each column of a DataFrame."""

from __future__ import annotations

__all__ = ["PlotColumnOutput"]


from arkas.content.plot_column import PlotColumnContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.dataframe import DataFrameState


class PlotColumnOutput(BaseStateOutput[DataFrameState]):
    r"""Implement an output to plot each column of a DataFrame.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import PlotColumnOutput
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.2, 4.2, 4.2, 2.2],
    ...         "col2": [1, 1, 1, 1],
    ...         "col3": [1, 2, 2, 2],
    ...     },
    ...     schema={"col1": pl.Float64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> output = PlotColumnOutput(DataFrameState(frame))
    >>> output
    PlotColumnOutput(
      (state): DataFrameState(dataframe=(4, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    PlotColumnContentGenerator(
      (state): DataFrameState(dataframe=(4, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        super().__init__(state)
        self._content = PlotColumnContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> PlotColumnContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
