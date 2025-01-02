r"""Implement a pairwise column co-occurrence analyzer."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceAnalyzer"]

import logging
from typing import TYPE_CHECKING

from grizz.utils.format import str_shape_diff

from arkas.analyzer.lazy import BaseInNLazyAnalyzer
from arkas.output.column_cooccurrence import ColumnCooccurrenceOutput

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

logger = logging.getLogger(__name__)


class ColumnCooccurrenceAnalyzer(BaseInNLazyAnalyzer):
    r"""Implement a pairwise column co-occurrence analyzer.

    Args:
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.analyzer import ColumnCooccurrenceAnalyzer
    >>> analyzer = ColumnCooccurrenceAnalyzer()
    >>> analyzer
    ColumnCooccurrenceAnalyzer(columns=None, exclude_columns=(), missing_policy='raise', ignore_self=False)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> output = analyzer.analyze(frame)
    >>> output
    ColumnCooccurrenceOutput(shape=(7, 3), ignore_self=False)

    ```
    """

    def __init__(
        self,
        columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = (),
        ignore_self: bool = False,
        missing_policy: str = "raise",
    ) -> None:
        super().__init__(
            columns=columns,
            exclude_columns=exclude_columns,
            missing_policy=missing_policy,
        )
        self._ignore_self = ignore_self

    def get_args(self) -> dict:
        return super().get_args() | {"ignore_self": self._ignore_self}

    def _analyze(self, frame: pl.DataFrame) -> ColumnCooccurrenceOutput:
        logger.info(
            "Analyzing the pairwise column co-occurrence of "
            f"{len(self.find_columns(frame)):,}..."
        )
        columns = self.find_common_columns(frame)
        out = frame.select(columns)
        logger.info(str_shape_diff(orig=frame.shape, final=out.shape))
        return ColumnCooccurrenceOutput(frame=out, ignore_self=self._ignore_self)
