r"""Contain the implementation of a HTML content generator that analyzes
the number of null values per column."""

from __future__ import annotations

__all__ = ["NullValueContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.figure.utils import figure2html
from arkas.plotter.null_value import NullValuePlotter

if TYPE_CHECKING:
    from arkas.state.null_value import NullValueState

logger = logging.getLogger(__name__)


class NullValueContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the number of null
    values per column.

    Args:
        state: The state containing the number of null values per
            column.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.content import NullValueContentGenerator
    >>> from arkas.state import NullValueState
    >>> content = NullValueContentGenerator(
    ...     NullValueState(
    ...         null_count=np.array([0, 1, 2]),
    ...         total_count=np.array([5, 5, 5]),
    ...         columns=["col1", "col2", "col3"],
    ...     )
    ... )
    >>> content
    NullValueContentGenerator(
      (state): NullValueState(num_columns=3, figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: NullValueState) -> None:
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
        ncols = len(self._state.columns)
        logger.info(f"Generating the null values bar plot for {ncols:,} columns...")
        figures = NullValuePlotter(state=self._state).plot()
        return Template(create_template()).render(
            {
                "ncols": f"{ncols:,}",
                "columns": ", ".join(self._state.columns),
                "figure": figure2html(figures["null_values"], close_fig=True),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.null_value import create_template
    >>> template = create_template()

    ```
    """
    return """This section analyzes the number and proportion of null values for the {{ncols}}
columns: <em>{{columns}}</em>.

<p>The columns are sorted by ascending order of number of null values in the following bar plot.

{{figure}}
"""
