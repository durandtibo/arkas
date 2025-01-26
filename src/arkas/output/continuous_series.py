r"""Implement an output to analyze a series with continuous values."""

from __future__ import annotations

__all__ = ["ContinuousSeriesOutput"]


from arkas.content.continuous_series import ContinuousSeriesContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.series import SeriesState


class ContinuousSeriesOutput(BaseStateOutput[SeriesState]):
    r"""Implement an output to analyze a series with continuous values.

    Args:
        state: The state containing the Series to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import ContinuousSeriesOutput
    >>> from arkas.state import SeriesState
    >>> output = ContinuousSeriesOutput(SeriesState(pl.Series("col1", [1, 2, 3, 4, 5, 6, 7])))
    >>> output
    ContinuousSeriesOutput(
      (state): SeriesState(name='col1', values=(7,), figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    ContinuousSeriesContentGenerator(
      (state): SeriesState(name='col1', values=(7,), figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: SeriesState) -> None:
        super().__init__(state)
        self._content = ContinuousSeriesContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> ContinuousSeriesContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
