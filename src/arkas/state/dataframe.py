r"""Implement the DataFrame state."""

from __future__ import annotations

__all__ = ["DataFrameState"]

import sys
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line, str_indent, str_mapping

from arkas.figure.utils import get_default_config
from arkas.metric.utils import check_nan_policy
from arkas.state.base import BaseState

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure.base import BaseFigureConfig


class DataFrameState(BaseState):
    r"""Implement the DataFrame state.

    Args:
        dataframe: The DataFrame.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.
        figure_config: An optional figure configuration.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [0, 0, 0, 0, 1, 1, 1],
    ...     }
    ... )
    >>> state = DataFrameState(frame)
    >>> state
    DataFrameState(dataframe=(7, 3), nan_policy='propagate', figure_config=MatplotlibFigureConfig())

    ```
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        nan_policy: str = "propagate",
        figure_config: BaseFigureConfig | None = None,
    ) -> None:
        self._dataframe = dataframe
        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy
        self._figure_config = figure_config or get_default_config()

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "dataframe": self._dataframe.shape,
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
                    "nan_policy": self._nan_policy,
                    "figure_config": self._figure_config,
                }
            )
        )
        return f"{self.__class__.__qualname__}({args})"

    @property
    def dataframe(self) -> pl.DataFrame:
        return self._dataframe

    @property
    def nan_policy(self) -> str:
        return self._nan_policy

    @property
    def figure_config(self) -> BaseFigureConfig | None:
        return self._figure_config

    def clone(self, deep: bool = True) -> Self:
        return self.__class__(
            dataframe=self._dataframe.clone() if deep else self._dataframe,
            nan_policy=self._nan_policy,
            figure_config=self._figure_config.clone() if deep else self._figure_config,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def get_args(self) -> dict:
        return {
            "dataframe": self._dataframe,
            "nan_policy": self._nan_policy,
            "figure_config": self._figure_config,
        }
