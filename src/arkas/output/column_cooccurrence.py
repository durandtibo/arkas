r"""Implement the pairwise column co-occurrence output."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceOutput"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.content.column_cooccurrence import ColumnCooccurrenceContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.lazy import BaseLazyOutput
from arkas.plotter.column_cooccurrence import ColumnCooccurrencePlotter

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure.base import BaseFigureConfig


class ColumnCooccurrenceOutput(BaseLazyOutput):
    r"""Implement the pairwise column co-occurrence output.

    Args:
        frame: The DataFrame to analyze.
        ignore_self: If ``True``, the diagonal of the co-occurrence
            matrix (a.k.a. self-co-occurrence) is set to 0.
        figure_config: The figure configuration.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.output import ColumnCooccurrenceOutput
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> output = ColumnCooccurrenceOutput(frame)
    >>> output
    ColumnCooccurrenceOutput(shape=(7, 3), ignore_self=False)
    >>> output.get_content_generator()
    ColumnCooccurrenceContentGenerator(shape=(7, 3), ignore_self=False)
    >>> output.get_evaluator()
    Evaluator(count=0)
    >>> output.get_plotter()
    ColumnCooccurrencePlotter(shape=(7, 3), ignore_self=False)

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

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self._ignore_self == other._ignore_self
            and objects_are_equal(self._frame, other._frame, equal_nan=equal_nan)
            and objects_are_equal(self._figure_config, other._figure_config, equal_nan=equal_nan)
        )

    def _get_content_generator(self) -> ColumnCooccurrenceContentGenerator:
        return ColumnCooccurrenceContentGenerator(
            frame=self._frame, ignore_self=self._ignore_self, figure_config=self._figure_config
        )

    def _get_evaluator(self) -> Evaluator:
        return Evaluator()

    def _get_plotter(self) -> ColumnCooccurrencePlotter:
        return ColumnCooccurrencePlotter(
            frame=self._frame, ignore_self=self._ignore_self, figure_config=self._figure_config
        )
