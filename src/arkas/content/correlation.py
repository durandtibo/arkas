r"""Contain the implementation of a HTML content generator that analyzes
the correlation between two columns."""

from __future__ import annotations

__all__ = [
    "CorrelationContentGenerator",
    "create_template",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.evaluator2.correlation import CorrelationEvaluator
from arkas.utils.dataframe import check_num_columns

if TYPE_CHECKING:

    from arkas.state.target_dataframe import DataFrameState

logger = logging.getLogger(__name__)


class CorrelationContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the correlation
    between two columns.

    Args:
        state: The state containing the DataFrame to analyze.
            The DataFrame must have only 2 columns, which are the two
            columns to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import CorrelationContentGenerator
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...     },
    ... )
    >>> content = CorrelationContentGenerator(DataFrameState(frame))
    >>> content
    CorrelationContentGenerator(
      (state): DataFrameState(dataframe=(7, 2), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        check_num_columns(state.dataframe, num_columns=2)
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
        xcol, ycol = self._state.dataframe.columns
        logger.info(f"Generating the correlation analysis between {xcol} and {ycol}...")
        metrics = CorrelationEvaluator(self._state).evaluate()

        return Template(create_template()).render(
            {
                "xcol": str(xcol),
                "ycol": str(ycol),
                "columns": ", ".join(self._state.dataframe.columns),
            }
            | metrics
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.correlation import create_template
    >>> template = create_template()

    ```
    """
    return """<p style="margin-top: 1rem;">
This section analyzes the correlation between <em>{{xcol}}</em> and <em>{{ycol}}</em>.
The correlation coefficient is a statistical measure of the strength of a
relationship between two variables. Its values can range from -1 to 1.
<ul>
  <li> A correlation coefficient of -1 describes a perfect negative, or inverse,
correlation, with values in one series rising as those in the other decline,
and vice versa. </li>
  <li> A coefficient of 1 shows a perfect positive correlation, or a direct relationship. </li>
  <li> A correlation coefficient of 0 means there is no direct relationship. </li>
</ul>
</p>

<ul>
  <li> <b>pearson coefficient</b>: {{pearson_coeff}} </li>
  <li> <b>pearson p-value</b>: {{pearson_pvalue}} </li>
  <li> <b>spearman coefficient</b>: {{spearman_coeff}} </li>
  <li> <b>spearman p-value</b>: {{spearman_pvalue}} </li>
  <li> <b>num samples</b>: {{count}} </li>
</ul>
"""
