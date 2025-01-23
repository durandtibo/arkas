r"""Contain the implementation of a HTML content generator that analyzes
the temporal distribution of a column with continuous values."""

from __future__ import annotations

__all__ = ["TemporalContinuousColumnContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.utils.dataframe import check_num_columns

if TYPE_CHECKING:
    from arkas.state.temporal_dataframe import TemporalDataFrameState


logger = logging.getLogger(__name__)


class TemporalContinuousColumnContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the temporal
    distribution of a column with continuous values.

    Args:
        state: The state containing the column to analyze.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.content import TemporalContinuouscolumnContentGenerator
    >>> from arkas.state import TemporalDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 2, 3],
    ...         "datetime": [
    ...             datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=2, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=3, day=3, tzinfo=timezone.utc),
    ...             datetime(year=2020, month=4, day=3, tzinfo=timezone.utc),
    ...         ],
    ...     },
    ...     schema={
    ...         "col1": pl.Int64,
    ...         "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
    ...     },
    ... )
    >>> content = TemporalContinuouscolumnContentGenerator(
    ...     TemporalDataFrameState(frame, temporal_column="datetime")
    ... )
    >>> content
    TemporalContinuouscolumnContentGenerator(
      (state): TemporalDataFrameState(dataframe=(4, 2), temporal_column='datetime', period=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: TemporalDataFrameState) -> None:
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
        first, second = self._state.dataframe.columns
        col = first
        if first == self._state.temporal_column:
            col = second
        logger.info(f"Generating the temporal continuous distribution analysis of {col!r}...")
        return Template(create_template()).render(
            {
                "column": col,
                "temporal_column": self._state.temporal_column,
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
    return """<p>This section analyzes the distribution of continuous values for column <em>{{column}}</em>.</p>
<ul>
  <li> <b>total values:</b> {{total_values}} </li>
  <li> <b>number of unique values:</b> {{unique_values}} </li>
  <li> <b>number of null values:</b> {{null_values}} / {{total_values}} ({{null_values_pct}}%) </li>
  <li> <b>range of values:</b> [{{min_value}}, {{max_value}}] </li>
  <li> <b>data type:</b> <em>{{dtype}}</em> </li>
</ul>

<p>The histogram shows the distribution of values in the range [{{xmin}}, {{xmax}}].</p>
{{figure}}

<details>
    <summary>[show statistics]</summary>
    <p style="margin-top: 1rem;">
    The following table shows some statistics about the distribution for column <em>{{column}}<em>.
    </p>
    {{table}}
</details>
"""
