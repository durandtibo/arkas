r"""Implement DataFrame state with a target column."""

from __future__ import annotations

__all__ = ["TargetDataFrameState"]

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


class TargetDataFrameState(DataFrameState):
    r"""Implement a DataFrame state with a target column.

    Args:
        dataframe: The DataFrame.
        target_column: The target column in the DataFrame.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.
        figure_config: An optional figure configuration.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.state import TargetDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ...     schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    ... )
    >>> state = TargetDataFrameState(frame, target_column="col3")
    >>> state
    TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())

    ```
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        target_column: str,
        nan_policy: str = "propagate",
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        super().__init__(dataframe=dataframe, nan_policy=nan_policy, figure_config=figure_config)

        check_column_exist(dataframe, target_column)
        self._target_column = target_column

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "dataframe": self._dataframe.shape,
                "target_column": self._target_column,
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
                    "target_column": self._target_column,
                    "nan_policy": self._nan_policy,
                    "figure_config": self._figure_config,
                }
            )
        )
        return f"{self.__class__.__qualname__}({args})"

    @property
    def target_column(self) -> str:
        return self._target_column

    def clone(self, deep: bool = True) -> Self:
        return self.__class__(
            dataframe=self._dataframe.clone() if deep else self._dataframe,
            target_column=self._target_column,
            nan_policy=self._nan_policy,
            figure_config=self._figure_config.clone() if deep else self._figure_config,
        )

    def get_args(self) -> dict:
        return super().get_args() | {"target_column": self._target_column}
