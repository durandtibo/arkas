r"""Contain the implementation of a HTML content generator that returns
the pairwise column co-occurrence."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceContentGenerator", "create_table", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from coola import objects_are_equal
from coola.utils import str_indent
from grizz.utils.cooccurrence import compute_pairwise_cooccurrence
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.figure.utils import figure2html
from arkas.plotter import ColumnCooccurrencePlotter

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

    from arkas.figure.base import BaseFigureConfig

logger = logging.getLogger(__name__)


class ColumnCooccurrenceContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that returns pairwise column co-
    occurrence.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.
        figure_config: The figure configuration.

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

    def __init__(
        self,
        frame: pl.DataFrame,
        ignore_self: bool = False,
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        self._frame = frame
        self._ignore_self = bool(ignore_self)
        self._figure_config = figure_config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(shape={self._frame.shape}, "
            f"ignore_self={self._ignore_self})"
        )

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
        return (
            self.ignore_self == other.ignore_self
            and objects_are_equal(self.frame, other.frame, equal_nan=equal_nan)
            and objects_are_equal(self._figure_config, other._figure_config, equal_nan=equal_nan)
        )

    def generate_content(self) -> str:
        logger.info("Generating the DataFrame summary content...")
        figures = ColumnCooccurrencePlotter(
            frame=self._frame, ignore_self=self._ignore_self, figure_config=self._figure_config
        ).plot()
        columns = list(self._frame.columns)
        return Template(create_template()).render(
            {
                "nrows": f"{self._frame.shape[0]:,}",
                "ncols": f"{self._frame.shape[1]:,}",
                "columns": ", ".join([f"{x!r}" for x in columns]),
                "figure": figure2html(figures["column_cooccurrence"], close_fig=True),
                "table": create_table_section(
                    matrix=compute_pairwise_cooccurrence(
                        frame=self._frame, ignore_self=self._ignore_self
                    ),
                    columns=columns,
                ),
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
    return """This section shows an analysis of the pairwise column co-occurrence.
<ul>
  <li> number of columns: {{ncols}} </li>
  <li> number of rows: {{nrows}}</li>
</ul>
{{figure}}
<details>
    <summary>[show columns]</summary>
    {{columns}}
</details>
<p style="margin-top: 1rem;">
{{table}}
"""


def create_table_section(matrix: np.ndarray, columns: Sequence[str], top: int = 50) -> str:
    r"""Return the HTML code of the table section.

    Args:
        matrix: The co-occurrence matrix.
        columns: The column names.
        top: The number of co-occurrence pairs to show in the table.

    Returns:
        The HTML code of the table section.

    Example usage:

    ```pycon

    >>> from arkas.content.column_cooccurrence import create_table_section
    >>> section = create_table_section(
    ...     matrix=np.array([[5, 7, 1], [0, 6, 3], [8, 2, 4]]), columns=["col1", "col2", "col3"]
    ... )

    ```
    """
    if matrix.shape[0] == 0:
        return "<span>&#9888;</span> No table is generated because the column is empty"

    return Template(
        """<details>
    <summary>[show top-{{top}} pairwise column co-occurrence]</summary>
    The following table shows the top-{{top}} pairwise column co-occurrences.
    <ul>
      <li> <b>rank</b>: is the rank of the co-occurrence </li>
      <li> <b>column (row)</b>: represents the first column of the co-occurrence matrix and corresponds to the row index </li>
      <li> <b>column (col)</b>: represents the second column of the co-occurrence matrix and corresponds to the column index </li>
      <li> <b>count</b>: is the number of co-occurrences </li>
      <li> <b>percentage</b>: is the percentage of co-occurrences w.r.t. the total of co-occurrences </li>
    </ul>

    {{table}}
</details>
"""
    ).render({"top": top, "table": create_table(matrix=matrix, columns=columns, top=top)})


def create_table(matrix: np.ndarray, columns: Sequence[str], top: int = 50) -> str:
    r"""Return the HTML code of the table.

    Args:
        matrix: The co-occurrence matrix.
        columns: The column names.
        top: The number of co-occurrence pairs to show in the table.

    Returns:
        The HTML code of the table.

    Example usage:

    ```pycon

    >>> from arkas.content.column_cooccurrence import create_table
    >>> section = create_table(
    ...     matrix=np.array([[5, 7, 1], [0, 6, 3], [8, 2, 4]]), columns=["col1", "col2", "col3"]
    ... )

    ```
    """
    total = matrix.sum().item()
    rows, cols = np.unravel_index(np.argsort(matrix, axis=None), matrix.shape)
    rows, cols = rows[: -top - 1 : -1], cols[: -top - 1 : -1]
    table_rows = []
    for i, (r, c) in enumerate(zip(rows, cols)):
        table_rows.append(
            create_table_row(
                rank=i + 1, col1=columns[r], col2=columns[c], count=matrix[r, c].item(), total=total
            )
        )
    table_rows = "\n".join(table_rows)
    return Template(
        """<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr><th>rank</th><th>column (row)</th><th>column (col)</th><th>count</th><th>percentage</th></tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
    ).render({"rows": str_indent(table_rows, num_spaces=8)})


def create_table_row(rank: int, col1: str, col2: str, count: int, total: int) -> str:
    r"""Return the HTML code of a table row.

    Args:
        rank: The rank of the pair of columns.
        col1: The first column.
        col2:  The second column.
        count: The number of co-occurrences.
        total: The total number of co-occurrences.

    Returns:
        The table row.

    Example usage:

    ```pycon

    >>> from arkas.content.column_cooccurrence import create_table_row
    >>> row = create_table_row(rank=2, col1="cat", col2="meow", count=42, total=100)

    ```
    """
    pct = 100 * count / total if total > 0 else float("nan")
    return Template(
        "<tr><th>{{rank}}</th>"
        "<td>{{col1}}</td>"
        "<td>{{col2}}</td>"
        '<td style="text-align: right;">{{count}}</td>'
        '<td style="text-align: right;">{{pct}}</td>'
        "</tr>"
    ).render(
        {
            "num_style": 'style="text-align: right;"',
            "rank": rank,
            "col1": col1,
            "col2": col2,
            "count": f"{count:,}",
            "pct": f"{pct:.4f} %",
        }
    )
