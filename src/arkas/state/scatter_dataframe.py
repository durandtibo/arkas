r"""Implement the DataFrame state for scatter plots."""

from __future__ import annotations

__all__ = ["ScatterDataFrameState"]

import sys
from typing import TYPE_CHECKING

from coola.utils.format import repr_mapping_line, str_indent, str_mapping

from arkas.state.dataframe import DataFrameState
from arkas.utils.dataframe import check_column_exist

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure.base import BaseFigureConfig


class ScatterDataFrameState(DataFrameState):
    r"""Implement the DataFrame state for scatter plots.

    Args:
        dataframe: The DataFrame.
        x: The x-axis data column.
        y: The y-axis data column.
        color: An optional color axis data column.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.
        figure_config: An optional figure configuration.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.state import ScatterDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> state = ScatterDataFrameState(frame, x="col1", y="col2")
    >>> state
    ScatterDataFrameState(dataframe=(7, 3), x='col1', y='col2', color=None, nan_policy='propagate', figure_config=MatplotlibFigureConfig())

    ```
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        nan_policy: str = "propagate",
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        super().__init__(dataframe=dataframe, nan_policy=nan_policy, figure_config=figure_config)

        check_column_exist(dataframe, x)
        check_column_exist(dataframe, y)
        if color is not None:
            check_column_exist(dataframe, color)
        self._x = x
        self._y = y
        self._color = color

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "dataframe": self._dataframe.shape,
                "x": self._x,
                "y": self._y,
                "color": self._color,
                "nan_policy": self._nan_policy,
                "figure_config": self._figure_config,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "dataframe": self._dataframe.shape,
                    "x": self._x,
                    "y": self._y,
                    "color": self._color,
                    "nan_policy": self._nan_policy,
                    "figure_config": self._figure_config,
                }
            )
        )
        return f"{self.__class__.__qualname__}({args})"

    @property
    def x(self) -> str:
        return self._x

    @property
    def y(self) -> str:
        return self._y

    @property
    def color(self) -> str | None:
        return self._color

    def clone(self, deep: bool = True) -> Self:
        return self.__class__(
            dataframe=self._dataframe.clone() if deep else self._dataframe,
            x=self._x,
            y=self._y,
            color=self._color,
            nan_policy=self._nan_policy,
            figure_config=self._figure_config.clone() if deep else self._figure_config,
        )

    def get_args(self) -> dict:
        return super().get_args() | {"x": self._x, "y": self._y, "color": self._color}
