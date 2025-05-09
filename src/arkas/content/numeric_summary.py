r"""Contain the implementation of a HTML content generator that
summarizes the numeric columns of a DataFrame."""

from __future__ import annotations

__all__ = [
    "NumericSummaryContentGenerator",
    "create_table",
    "create_table_quantiles",
    "create_table_quantiles_row",
    "create_table_row",
    "create_template",
]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.content.utils import float_to_str
from arkas.evaluator2.numeric_stats import NumericStatisticsEvaluator
from arkas.utils.style import get_tab_number_style

if TYPE_CHECKING:
    from arkas.state.dataframe import DataFrameState

logger = logging.getLogger(__name__)


class NumericSummaryContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that summarizes the numeric columns
    of a DataFrame.

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
      (state): DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
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
        metrics = NumericStatisticsEvaluator(self._state).evaluate()
        return Template(create_template()).render(
            {
                "nrows": f"{nrows:,}",
                "ncols": f"{ncols:,}",
                "columns": ", ".join(self._state.dataframe.columns),
                "table": create_table(metrics),
                "table_quantiles": create_table_quantiles(metrics),
            }
        )


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.numeric_summary import create_template
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

<details>
    <summary>[show additional statistics]</summary>

    <p style="margin-top: 1rem;">
    The following table shows some quantiles for each column. </p>

    {{table_quantiles}}
</details>
"""


def create_table(
    col_metrics: dict[str, dict[str, float]],
) -> str:
    r"""Return a HTML representation of a table with some statisticts
    about each column.

    Args:
        col_metrics: The dictionary of metrics for each column.

    Returns:
        The HTML representation of the table.

    Example usage:

    ```pycon

    >>> from arkas.content.numeric_summary import create_table
    >>> row = create_table(
    ...     {
    ...         "col1": {
    ...             "count": 7,
    ...             "nunique": 7,
    ...             "num_nans": 0,
    ...             "num_nulls": 0,
    ...             "mean": 4.0,
    ...             "std": 2.0,
    ...             "skewness": 0.0,
    ...             "kurtosis": -1.25,
    ...             "min": 1.0,
    ...             "q001": 1.006,
    ...             "q01": 1.06,
    ...             "q05": 1.3,
    ...             "q10": 1.6,
    ...             "q25": 2.5,
    ...             "median": 4.0,
    ...             "q75": 5.5,
    ...             "q90": 6.4,
    ...             "q95": 6.7,
    ...             "q99": 6.94,
    ...             "q999": 6.994,
    ...             "max": 7.0,
    ...             ">0": 7,
    ...             "<0": 0,
    ...             "=0": 0,
    ...         }
    ...     }
    ... )

    ```
    """
    rows = "\n".join(
        [create_table_row(column, metrics=metrics) for column, metrics in col_metrics.items()]
    )
    return Template(
        """<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
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


def create_table_row(column: str, metrics: dict[str, float]) -> str:
    r"""Create the HTML code of a new table row.

    Args:
        column: The column name.
        metrics: The dictionary of metrics.

    Returns:
        The HTML code of a row.

    Example usage:

    ```pycon

    >>> from arkas.content.numeric_summary import create_table_row
    >>> row = create_table_row(
    ...     column="col",
    ...     metrics={
    ...         "count": 101,
    ...         "num_nulls": 0,
    ...         "num_nans": 0,
    ...         "nunique": 101,
    ...         "mean": 50.0,
    ...         "std": 29.0,
    ...         "skewness": 0.0,
    ...         "kurtosis": -1.2,
    ...         "min": 0.0,
    ...         "q001": 0.1,
    ...         "q01": 1.0,
    ...         "q05": 5.0,
    ...         "q10": 10.0,
    ...         "q25": 25.0,
    ...         "median": 50.0,
    ...         "q75": 75.0,
    ...         "q90": 90.0,
    ...         "q95": 95.0,
    ...         "q99": 99.0,
    ...         "q999": 99.9,
    ...         "max": 100.0,
    ...         ">0": 100,
    ...         "<0": 0,
    ...         "=0": 1,
    ...     },
    ... )

    ```
    """
    nan = metrics["num_nans"]
    null = metrics["num_nulls"]
    nunique = metrics["nunique"]
    total = metrics["count"]
    negative = metrics["<0"]
    zero = metrics["=0"]
    positive = metrics[">0"]
    return Template(
        """<tr>
    <th>{{column}}</th>
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
            "num_style": f'style="{get_tab_number_style()}"',
            "column": column,
            "null": f"{null:,} ({100 * null / total if total else float('nan'):.2f}%)",
            "nan": f"{nan:,} ({100 * nan / total if total else float('nan'):.2f}%)",
            "nunique": f"{nunique:,} ({100 * nunique / total if total else float('nan'):.2f}%)",
            "mean": float_to_str(metrics["mean"]),
            "std": float_to_str(metrics["std"]),
            "skewness": float_to_str(metrics["skewness"]),
            "kurtosis": float_to_str(metrics["kurtosis"]),
            "min": float_to_str(metrics["min"]),
            "median": float_to_str(metrics["median"]),
            "max": float_to_str(metrics["max"]),
            "negative": f"{negative:,} ({100 * negative / total if total else float('nan'):.2f}%)",
            "zero": f"{zero:,} ({100 * zero / total if total else float('nan'):.2f}%)",
            "positive": f"{positive:,} ({100 * positive / total if total else float('nan'):.2f}%)",
        }
    )


def create_table_quantiles(
    col_metrics: dict[str, dict[str, float]],
) -> str:
    r"""Return a HTML representation of a table with quantile statisticts
    for each column.

    Args:
        col_metrics: The dictionary of metrics for each column.

    Returns:
        The HTML representation of the table.

    Example usage:

    ```pycon

    >>> from arkas.content.numeric_summary import create_table
    >>> row = create_table(
    ...     {
    ...         "col1": {
    ...             "count": 7,
    ...             "nunique": 7,
    ...             "num_nans": 0,
    ...             "num_nulls": 0,
    ...             "mean": 4.0,
    ...             "std": 2.0,
    ...             "skewness": 0.0,
    ...             "kurtosis": -1.25,
    ...             "min": 1.0,
    ...             "q001": 1.006,
    ...             "q01": 1.06,
    ...             "q05": 1.3,
    ...             "q10": 1.6,
    ...             "q25": 2.5,
    ...             "median": 4.0,
    ...             "q75": 5.5,
    ...             "q90": 6.4,
    ...             "q95": 6.7,
    ...             "q99": 6.94,
    ...             "q999": 6.994,
    ...             "max": 7.0,
    ...             ">0": 7,
    ...             "<0": 0,
    ...             "=0": 0,
    ...         }
    ...     }
    ... )

    ```
    """
    rows = "\n".join(
        [
            create_table_quantiles_row(column=column, metrics=metrics)
            for column, metrics in col_metrics.items()
        ]
    )
    return Template(
        """<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>min</th>
            <th>q0.001</th>
            <th>q0.01</th>
            <th>q0.05</th>
            <th>q0.10</th>
            <th>q0.25</th>
            <th>median</th>
            <th>q0.75</th>
            <th>q0.90</th>
            <th>q0.95</th>
            <th>q0.99</th>
            <th>q0.999</th>
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


def create_table_quantiles_row(column: str, metrics: dict[str, float]) -> str:
    r"""Create the HTML code of a new table row.

    Args:
        column: The column name.
        metrics: The dictionary of metrics.

    Returns:
        The HTML code of a row.

    Example usage:

    ```pycon

    >>> from arkas.content.numeric_summary import create_table_row
    >>> row = create_table_row(
    ...     column="col",
    ...     metrics={
    ...         "count": 101,
    ...         "num_nulls": 0,
    ...         "num_nans": 0,
    ...         "nunique": 101,
    ...         "mean": 50.0,
    ...         "std": 29.0,
    ...         "skewness": 0.0,
    ...         "kurtosis": -1.2,
    ...         "min": 0.0,
    ...         "q001": 0.1,
    ...         "q01": 1.0,
    ...         "q05": 5.0,
    ...         "q10": 10.0,
    ...         "q25": 25.0,
    ...         "median": 50.0,
    ...         "q75": 75.0,
    ...         "q90": 90.0,
    ...         "q95": 95.0,
    ...         "q99": 99.0,
    ...         "q999": 99.9,
    ...         "max": 100.0,
    ...         ">0": 100,
    ...         "<0": 0,
    ...         "=0": 1,
    ...     },
    ... )

    ```
    """
    return Template(
        """<tr>
    <th>{{column}}</th>
    <td {{num_style}}>{{min}}</td>
    <td {{num_style}}>{{q001}}</td>
    <td {{num_style}}>{{q01}}</td>
    <td {{num_style}}>{{q05}}</td>
    <td {{num_style}}>{{q10}}</td>
    <td {{num_style}}>{{q25}}</td>
    <td {{num_style}}>{{median}}</td>
    <td {{num_style}}>{{q75}}</td>
    <td {{num_style}}>{{q90}}</td>
    <td {{num_style}}>{{q95}}</td>
    <td {{num_style}}>{{q99}}</td>
    <td {{num_style}}>{{q999}}</td>
    <td {{num_style}}>{{max}}</td>
</tr>"""
    ).render(
        {
            "num_style": f'style="{get_tab_number_style()}"',
            "column": column,
            "min": float_to_str(metrics["min"]),
            "q001": float_to_str(metrics["q001"]),
            "q01": float_to_str(metrics["q01"]),
            "q05": float_to_str(metrics["q05"]),
            "q10": float_to_str(metrics["q10"]),
            "q25": float_to_str(metrics["q25"]),
            "median": float_to_str(metrics["median"]),
            "q75": float_to_str(metrics["q75"]),
            "q90": float_to_str(metrics["q90"]),
            "q95": float_to_str(metrics["q95"]),
            "q99": float_to_str(metrics["q99"]),
            "q999": float_to_str(metrics["q999"]),
            "max": float_to_str(metrics["max"]),
        }
    )
