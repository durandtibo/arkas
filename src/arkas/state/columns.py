r"""Implement DataFrame states with a target columns."""

from __future__ import annotations

__all__ = ["TwoColumnDataFrameState"]

from typing import TYPE_CHECKING, Any

from arkas.state.dataframe import DataFrameState
from arkas.utils.dataframe import check_column_exist

if TYPE_CHECKING:
    import polars as pl

    from arkas.figure.base import BaseFigureConfig


class TwoColumnDataFrameState(DataFrameState):
    r"""Implement a DataFrame state with a target column.

    Args:
        dataframe: The DataFrame.
        column1: The first target column in the DataFrame.
        column2: The second target column in the DataFrame.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.
        figure_config: An optional figure configuration.
        **kwargs: Additional keyword arguments.

    Example usage:

    ```pycon

    >>> from datetime import datetime, timezone
    >>> import polars as pl
    >>> from arkas.state import TwoColumnDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [0, 1, 1, 0, 0, 1, 0],
    ...         "col2": [0, 1, 0, 1, 0, 1, 0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ...     schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    ... )
    >>> state = TwoColumnDataFrameState(frame, column1="col3", column2="col1")
    >>> state
    TwoColumnDataFrameState(dataframe=(7, 3), column1='col3', column2='col1', nan_policy='propagate', figure_config=MatplotlibFigureConfig())

    ```
    """

    def __init__(
        self,
        dataframe: pl.DataFrame,
        column1: str,
        column2: str,
        nan_policy: str = "propagate",
        figure_config: BaseFigureConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataframe=dataframe,
            nan_policy=nan_policy,
            figure_config=figure_config,
            **kwargs,
        )
        check_column_exist(dataframe, column1)
        self._column1 = column1
        check_column_exist(dataframe, column2)
        self._column2 = column2

    @property
    def column1(self) -> str:
        return self._column1

    @property
    def column2(self) -> str:
        return self._column2

    def get_args(self) -> dict:
        args = super().get_args()
        return {
            "dataframe": args.pop("dataframe"),
            "column1": self._column1,
            "column2": self._column2,
        } | args
