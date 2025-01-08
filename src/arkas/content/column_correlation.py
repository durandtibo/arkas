r"""Contain the implementation of a HTML content generator that analyzes
the correlation between 1 target column and other columns."""

from __future__ import annotations

__all__ = [
    "ColumnCorrelationContentGenerator",
    "create_table",
    "create_table_row",
    "create_template",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.content.utils import float_to_str
from arkas.utils.stats import compute_statistics_continuous

if TYPE_CHECKING:
    import polars as pl

    from arkas.state.target_dataframe import TargetDataFrameState

logger = logging.getLogger(__name__)


class ColumnCorrelationContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the correlation
    between 1 target column and other columns.

    Args:
        state: The state containing the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import ColumnCorrelationContentGenerator
    >>> from arkas.state import TargetDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ... )
    >>> content = ColumnCorrelationContentGenerator(
    ...     TargetDataFrameState(frame, target_column="col3")
    ... )
    >>> content
    ColumnCorrelationContentGenerator(
      (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: TargetDataFrameState) -> None:
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
        logger.info(
            f"Generating the correlation analysis between {self._state.target_column} and {list(self._state.dataframe.columns)}..."
        )
        return Template(create_template()).render(
            {
                "columns": ", ".join(self._state.dataframe.columns),
                "table": create_table(self._state.dataframe),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.column_correlation import create_template
    >>> template = create_template()

    ```
    """
    return """This section shows a short summary of each column.

<ul>
  <li> <b>column</b>: is the column name</li>
  <li> <b>dtype</b>: is the column data type </li>
  <li> <b>null</b>: is the number (and percentage) of null values in the column </li>
  <li> <b>nan</b>: is the number (and percentage) of not a number (NaN) values in the column </li>
  <li> <b>unique</b>: is the number (and percentage) of unique values in the column </li>
  <li> <b>negative</b>: is the number (and percentage) of strictly negative values (<span>&#60;</span>0) in the column </li>
  <li> <b>zero</b>: is the number (and percentage) of zero values (=0) in the column </li>
  <li> <b>positive</b>: is the number (and percentage) of strictly positive values (<span>&#62;</span>0) in the column </li>
</ul>

<p style="margin-top: 1rem;">
<b>General statistics about the DataFrame</b>
{{table}}
"""


def create_table(
    frame: pl.DataFrame,
) -> str:
    r"""Return a HTML representation of a table with some statisticts
    about each column.

    Args:
        frame: The DataFrame to analyze.

    Returns:
        The HTML representation of the table.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content.column_correlation import create_table
    >>> dataframe = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ... )
    >>> row = create_table(dataframe)

    ```
    """
    rows = "\n".join([create_table_row(series=series) for series in frame])
    return Template(
        """<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>dtype</th>
            <th>null</th>
            <th>nan</th>
            <th>unique</th>
            <th>negative</th>
            <th>zero</th>
            <th>positive</th>
            <th>mean</th>
            <th>std</th>
            <th>skewness</th>
            <th>kurtosis</th>
            <th>min</th>
            <th>median</th>
            <th>max</th>
        </tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
    ).render({"rows": rows})


def create_table_row(series: pl.Series) -> str:
    r"""Create the HTML code of a new table row.

    Args:
        series: The series to analyze.

    Returns:
        The HTML code of a row.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content.column_correlation import create_table_row
    >>> row = create_table_row(pl.Series("col1", [1, 2, 3, 4, 5, 6, 7]))

    ```
    """
    stats = compute_statistics_continuous(series)
    nan = int(series.is_nan().sum())
    null = stats["num_nulls"]
    nunique = stats["nunique"]
    total = stats["count"]
    negative = stats["<0"]
    zero = stats["=0"]
    positive = stats[">0"]
    return Template(
        """<tr>
    <th>{{column}}</th>
    <td>{{dtype}}</td>
    <td {{num_style}}>{{null}}</td>
    <td {{num_style}}>{{nan}}</td>
    <td {{num_style}}>{{nunique}}</td>
    <td {{num_style}}>{{negative}}</td>
    <td {{num_style}}>{{zero}}</td>
    <td {{num_style}}>{{positive}}</td>
    <td {{num_style}}>{{mean}}</td>
    <td {{num_style}}>{{std}}</td>
    <td {{num_style}}>{{skewness}}</td>
    <td {{num_style}}>{{kurtosis}}</td>
    <td {{num_style}}>{{min}}</td>
    <td {{num_style}}>{{median}}</td>
    <td {{num_style}}>{{max}}</td>
</tr>"""
    ).render(
        {
            "num_style": 'style="text-align: right;"',
            "column": series.name,
            "dtype": series.dtype,
            "null": f"{null:,} ({100 * null / total if total else float('nan'):.2f}%)",
            "nan": f"{nan:,} ({100 * nan / total if total else float('nan'):.2f}%)",
            "nunique": f"{nunique:,} ({100 * nunique / total if total else float('nan'):.2f}%)",
            "mean": float_to_str(stats["mean"]),
            "std": float_to_str(stats["std"]),
            "skewness": float_to_str(stats["skewness"]),
            "kurtosis": float_to_str(stats["kurtosis"]),
            "min": float_to_str(stats["min"]),
            "median": float_to_str(stats["median"]),
            "max": float_to_str(stats["max"]),
            "negative": f"{negative:,} ({100 * negative / total if total else float('nan'):.2f}%)",
            "zero": f"{zero:,} ({100 * zero / total if total else float('nan'):.2f}%)",
            "positive": f"{positive:,} ({100 * positive / total if total else float('nan'):.2f}%)",
        }
    )
