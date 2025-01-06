r"""Contain the implementation of a HTML content generator that analyzes
a Series with continuous values."""

from __future__ import annotations

__all__ = ["ContinuousSeriesContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.figure.utils import figure2html
from arkas.plotter.continuous_series import ContinuousSeriesPlotter

if TYPE_CHECKING:
    from arkas.state.series import SeriesState


logger = logging.getLogger(__name__)


class ContinuousSeriesContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes a Series with
    continuous values.

    Args:
        state: The state containing the Series to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import ContinuousSeriesContentGenerator
    >>> from arkas.state import SeriesState
    >>> content = ContinuousSeriesContentGenerator(
    ...     SeriesState(pl.Series("col1", [1, 2, 3, 4, 5, 6, 7]))
    ... )
    >>> content
    ContinuousSeriesContentGenerator(
      (state): SeriesState(name='col1', values=(7,), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: SeriesState) -> None:
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

    def generate_content(self) -> str:
        logger.info(f"Generating the continuous distribution of {self._state.series.name}...")
        figures = ContinuousSeriesPlotter(state=self._state).plot()
        return Template(create_template()).render(
            {
                "figure": figure2html(figures["continuous_histogram"], close_fig=True),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.continuous_series import create_template
    >>> template = create_template()

    ```
    """
    return """This section plots the content of some columns.
The x-axis is the row index and the y-axis shows the value.

{{figure}}
"""
