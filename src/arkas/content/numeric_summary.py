r"""Contain the implementation of a HTML content generator that plots
the content of each column."""

from __future__ import annotations

__all__ = ["NumericSummaryContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator

if TYPE_CHECKING:
    from arkas.state.dataframe import DataFrameState


logger = logging.getLogger(__name__)


class NumericSummaryContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that summarizes some numeric
    columns.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import NumericSummaryContentGenerator
    >>> from arkas.state import DataFrameState
    >>> dataframe = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     }
    ... )
    >>> content = NumericSummaryContentGenerator(DataFrameState(dataframe))
    >>> content
    NumericSummaryContentGenerator(
      (state): DataFrameState(dataframe=(7, 3), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
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
        nrows, ncols = self._state.dataframe.shape
        logger.info(f"Generating the summary of {ncols:,} numeric columns...")
        return Template(create_template()).render(
            {
                "nrows": f"{nrows:,}",
                "ncols": f"{ncols:,}",
                "columns": ", ".join(self._state.dataframe.columns),
                # "figure": figure2html(figures["plot_column"], close_fig=True),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.plot_column import create_template
    >>> template = create_template()

    ```
    """
    return """This section plots the content of some columns.
<ul>
  <li> {{ncols}} columns: {{columns}} </li>
  <li> number of rows: {{nrows}}</li>
</ul>
{{figure}}
"""
