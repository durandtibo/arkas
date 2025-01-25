r"""Contain the implementation of a HTML content generator that analyzes
the correlation between 1 target column and other columns."""

from __future__ import annotations

__all__ = [
    "ColumnCorrelationContentGenerator",
    "create_table",
    "create_table_row",
    "create_template",
    "sort_metrics",
]

import logging
import math
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.evaluator2.column_correlation import ColumnCorrelationEvaluator
from arkas.utils.style import get_tab_number_style

if TYPE_CHECKING:
    from arkas.state.target_dataframe import TargetDataFrameState

logger = logging.getLogger(__name__)


class ColumnCorrelationContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the correlation
    between 1 target column and other columns.

    Args:
        evaluator: The evaluator object to compute the correlations.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import ColumnCorrelationContentGenerator
    >>> from arkas.evaluator2 import ColumnCorrelationEvaluator
    >>> from arkas.state import TargetDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ... )
    >>> content = ColumnCorrelationContentGenerator(
    ...     ColumnCorrelationEvaluator(TargetDataFrameState(frame, target_column="col3"))
    ... )
    >>> content
    ColumnCorrelationContentGenerator(
      (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, evaluator: ColumnCorrelationEvaluator) -> None:
        self._evaluator = evaluator

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._evaluator.state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._evaluator.state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._evaluator.equal(other._evaluator, equal_nan=equal_nan)

    def generate_content(self) -> str:
        state = self._evaluator.state
        logger.info(
            f"Generating the correlation analysis between {state.target_column} "
            f"and {list(state.dataframe.columns)}..."
        )
        metrics = self._evaluator.evaluate()
        metrics = sort_metrics(metrics, key=state.get_arg("sort_metric", "spearman_coeff"))
        columns = list(state.dataframe.columns)
        columns.remove(state.target_column)
        nrows, ncols = state.dataframe.shape
        return Template(create_template()).render(
            {
                "nrows": f"{nrows:,}",
                "ncols": f"{ncols:,}",
                "columns": ", ".join(state.dataframe.columns),
                "table": create_table(metrics),
                "target_column": f"{state.target_column}",
            }
        )

    @classmethod
    def from_state(cls, state: TargetDataFrameState) -> ColumnCorrelationContentGenerator:
        r"""Instantiate a ``ColumnCorrelationContentGenerator`` object
        from a state.

        Args:
            state: The state with the data to analyze.

        Returns:
            The instantiated object.

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
        >>> content = ColumnCorrelationContentGenerator.from_state(
        ...     TargetDataFrameState(frame, target_column="col3")
        ... )
        >>> content
        ColumnCorrelationContentGenerator(
          (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
        )

        ```
        """
        return cls(ColumnCorrelationEvaluator(state))


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
    return """<p style="margin-top: 1rem;">
This section analyzes the correlation between <em>{{target_column}}</em> and other columns.
The correlation coefficient is a statistical measure of the strength of a
relationship between two variables. Its values can range from -1 to 1.
<ul>
  <li> A correlation coefficient of -1 describes a perfect negative, or inverse,
correlation, with values in one series rising as those in the other decline,
and vice versa. </li>
  <li> A coefficient of 1 shows a perfect positive correlation, or a direct relationship. </li>
  <li> A correlation coefficient of 0 means there is no direct relationship. </li>
</ul>
The DataFrame has {{nrows}} rows and {{ncols}} columns.
</p>

{{table}}
"""


def create_table(metrics: dict[str, dict]) -> str:
    r"""Return a HTML representation of a table with some statisticts
    about each column.

    Args:
        metrics: The dictionary of metrics.

    Returns:
        The HTML representation of the table.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content.column_correlation import create_table
    >>> row = create_table(
    ...     metrics={
    ...         "col1": {
    ...             "count": 7,
    ...             "pearson_coeff": 1.0,
    ...             "pearson_pvalue": 0.0,
    ...             "spearman_coeff": 1.0,
    ...             "spearman_pvalue": 0.0,
    ...         },
    ...         "col2": {
    ...             "count": 7,
    ...             "pearson_coeff": -1.0,
    ...             "pearson_pvalue": 0.0,
    ...             "spearman_coeff": -1.0,
    ...             "spearman_pvalue": 0.0,
    ...         },
    ...     },
    ... )

    ```
    """
    rows = "\n".join(
        [create_table_row(column=col, metrics=values) for col, values in metrics.items()]
    )
    return Template(
        """<table class="table table-hover table-responsive w-auto" >
    <thead class="thead table-group-divider">
        <tr>
            <th>column</th>
            <th>num samples</th>
            <th>pearson coefficient (p-value)</th>
            <th>spearman coefficient (p-value)</th>
        </tr>
    </thead>
    <tbody class="tbody table-group-divider">
        {{rows}}
        <tr class="table-group-divider"></tr>
    </tbody>
</table>
"""
    ).render({"rows": rows})


def create_table_row(column: str, metrics: dict) -> str:
    r"""Create the HTML code of a new table row.

    Args:
        column: The column name
        metrics: The dictionary of metrics with the correlation scores.

    Returns:
        The HTML code of a row.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content.column_correlation import create_table_row
    >>> row = create_table_row(
    ...     column="col1",
    ...     metrics={
    ...         "count": 7,
    ...         "pearson_coeff": 1.0,
    ...         "pearson_pvalue": 0.0,
    ...         "spearman_coeff": 1.0,
    ...         "spearman_pvalue": 0.0,
    ...     },
    ... )

    ```
    """
    return Template(
        """<tr>
    <th>{{column}}</th>
    <td {{num_style}}>{{count}}</td>
    <td {{num_style}}>{{pearson_coeff}} ({{pearson_pvalue}})</td>
    <td {{num_style}}>{{spearman_coeff}} ({{spearman_pvalue}})</td>
</tr>"""
    ).render(
        {
            "num_style": f'style="{get_tab_number_style()}"',
            "column": column,
            "count": f'{metrics.get("count", 0):,}',
            "pearson_coeff": f'{metrics.get("pearson_coeff", float("nan")):.4f}',
            "pearson_pvalue": f'{metrics.get("pearson_pvalue", float("nan")):.4f}',
            "spearman_coeff": f'{metrics.get("spearman_coeff", float("nan")):.4f}',
            "spearman_pvalue": f'{metrics.get("spearman_pvalue", float("nan")):.4f}',
        }
    )


def sort_metrics(
    metrics: dict[str, dict[str, float]], key: str = "spearman_coeff"
) -> dict[str, dict[str, float]]:
    r"""Sort the dictionary of metrics by a given key.

    Args:
        metrics: The dictionary of metrics to sort.
        key: The key to use to sort the metrics.

    Returns:
        The sorted dictionary of metrics.
    """

    def get_metric(item: Any) -> float:
        val = item[1][key]
        if math.isnan(val):
            val = float("-inf")
        return val

    return dict(sorted(metrics.items(), key=get_metric, reverse=True))
