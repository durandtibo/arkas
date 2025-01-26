r"""Implement an output to scatter plot some columns."""

from __future__ import annotations

__all__ = ["ScatterColumnOutput"]


from arkas.content.scatter_column import ScatterColumnContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.scatter_dataframe import ScatterDataFrameState


class ScatterColumnOutput(BaseStateOutput[ScatterDataFrameState]):
    r"""Implement an output to scatter plot some columns.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import ScatterColumnOutput
    >>> from arkas.state import ScatterDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0],
    ...         "col2": [0, 1, 0, 1],
    ...         "col3": [1, 0, 0, 0],
    ...     },
    ...     schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> output = ScatterColumnOutput(ScatterDataFrameState(frame, x="col1", y="col2"))
    >>> output
    ScatterColumnOutput(
      (state): ScatterDataFrameState(dataframe=(4, 3), x='col1', y='col2', color=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    ScatterColumnContentGenerator(
      (state): ScatterDataFrameState(dataframe=(4, 3), x='col1', y='col2', color=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: ScatterDataFrameState) -> None:
        super().__init__(state)
        self._content = ScatterColumnContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> ScatterColumnContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
