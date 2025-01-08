r"""Implement an analyzer that generates a summary of the numeric
columns of a DataFrame."""

from __future__ import annotations

__all__ = ["NumericSummaryAnalyzer"]

import logging
from typing import TYPE_CHECKING

from grizz.utils.format import str_shape_diff
from polars import selectors as cs

from arkas.analyzer.lazy import BaseInNLazyAnalyzer
from arkas.output.frame_summary import DataFrameSummaryOutput

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class NumericSummaryAnalyzer(BaseInNLazyAnalyzer):
    r"""Implement an analyzer to show a summary of the numeric columns of
    a DataFrame.

    Args:
        columns: The columns to analyze. If ``None``, it analyzes all
            the columns.
        exclude_columns: The columns to exclude from the input
            ``columns``. If any column is not found, it will be ignored
            during the filtering process.
        missing_policy: The policy on how to handle missing columns.
            The following options are available: ``'ignore'``,
            ``'warn'``, and ``'raise'``. If ``'raise'``, an exception
            is raised if at least one column is missing.
            If ``'warn'``, a warning is raised if at least one column
            is missing and the missing columns are ignored.
            If ``'ignore'``, the missing columns are ignored and
            no warning message appears.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.analyzer import NumericSummaryAnalyzer
    >>> analyzer = NumericSummaryAnalyzer()
    >>> analyzer
    NumericSummaryAnalyzer(columns=None, exclude_columns=(), missing_policy='raise')
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 0, 1],
    ...         "col2": [1, 0, 1, 0],
    ...         "col3": [1, 1, 1, 1],
    ...     },
    ...     schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
    ... )
    >>> output = analyzer.analyze(frame)
    >>> output
    DataFrameSummaryOutput(shape=(4, 3), top=5)

    ```
    """

    def _analyze(self, frame: pl.DataFrame) -> DataFrameSummaryOutput:
        logger.info("Analyzing the numeric columns...")
        columns = self.find_common_columns(frame)
        out = frame.select(cs.by_name(columns) & cs.numeric())
        logger.info(str_shape_diff(orig=frame.shape, final=out.shape))
        return DataFrameSummaryOutput(frame=frame)
