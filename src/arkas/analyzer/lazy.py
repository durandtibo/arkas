r"""Define a base class to implement a lazy analyzer."""

from __future__ import annotations

__all__ = ["BaseLazyAnalyzer"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from arkas.analyzer.base import BaseAnalyzer

if TYPE_CHECKING:
    import polars as pl

    from arkas.output import BaseOutput

logger = logging.getLogger(__name__)


class BaseLazyAnalyzer(BaseAnalyzer):
    r"""Define a base class to implement a lazy analyzer.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.analyzer import DataFrameSummaryAnalyzer
    >>> analyzer = DataFrameSummaryAnalyzer()
    >>> analyzer
    DataFrameSummaryAnalyzer(top=5, sort=False)
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

    def analyze(self, frame: pl.DataFrame, lazy: bool = True) -> BaseOutput:
        output = self._analyze(frame)
        if not lazy:
            output = output.compute()
        return output

    @abstractmethod
    def _analyze(self, frame: pl.DataFrame) -> BaseOutput:
        r"""Analyze the DataFrame.

        Args:
            frame: The DataFrame to analyze.

        Returns:
            The generated output.
        """
