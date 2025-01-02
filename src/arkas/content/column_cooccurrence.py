r"""Contain the implementation of a HTML content generator that returns
the pairwise column co-occurrence."""

from __future__ import annotations

__all__ = [
    "ColumnCooccurrenceContentGenerator",
    "create_template",
]

import logging
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.plotter import ColumnCooccurrencePlotter
from arkas.utils.figure import figure2html

if TYPE_CHECKING:

    import polars as pl

logger = logging.getLogger(__name__)


class ColumnCooccurrenceContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that returns pairwise column co-
    occurrence.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import ColumnCooccurrenceContentGenerator
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> content = ColumnCooccurrenceContentGenerator(frame)
    >>> content
    ColumnCooccurrenceContentGenerator(shape=(7, 3), ignore_self=False)

    ```
    """

    def __init__(self, frame: pl.DataFrame, ignore_self: bool = False) -> None:
        self._frame = frame
        self._ignore_self = bool(ignore_self)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(shape={self._frame.shape}, ignore_self={self._ignore_self})"

    @property
    def frame(self) -> pl.DataFrame:
        r"""The DataFrame to analyze."""
        return self._frame

    @property
    def ignore_self(self) -> bool:
        return self._ignore_self

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.ignore_self == other.ignore_self and objects_are_equal(
            self.frame, other.frame, equal_nan=equal_nan
        )

    def generate_content(self) -> str:
        logger.info("Generating the DataFrame summary content...")
        figures = ColumnCooccurrencePlotter(frame=self._frame, ignore_self=self._ignore_self).plot()
        return Template(create_template()).render(
            {
                "nrows": f"{self._frame.shape[0]:,}",
                "ncols": f"{self._frame.shape[1]:,}",
                "figure": figure2html(figures["column_cooccurrence"], close_fig=True),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.frame_summary import create_template
    >>> template = create_template()

    ```
    """
    return """This section shows the pairwise column co-occurrence.
<ul>
  <li> number of columns: {{ncols}} </li>
  <li> number of rows: {{nrows}}</li>
</ul>
{{figure}}
"""
