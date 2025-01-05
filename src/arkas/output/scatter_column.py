r"""Implement an output to scatter plot some columns."""

from __future__ import annotations

__all__ = ["ScatterColumnOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.content.vanilla import ContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.lazy import BaseLazyOutput
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    from arkas.state.temporal_dataframe import ScatterDataFrameState


class ScatterColumnOutput(BaseLazyOutput):
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
      (state): ScatterDataFrameState(dataframe=(4, 3), x='col1', y='col2', color=None, figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    ContentGenerator()
    >>> output.get_evaluator()
    Evaluator(count=0)
    >>> output.get_plotter()
    Plotter(count=0)

    ```
    """

    def __init__(self, state: ScatterDataFrameState) -> None:
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)

    def _get_content_generator(self) -> ContentGenerator:
        return ContentGenerator()

    def _get_evaluator(self) -> Evaluator:
        return Evaluator()

    def _get_plotter(self) -> Plotter:
        return Plotter()
