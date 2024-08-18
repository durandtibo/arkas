r"""Contain a DataFrame ingestor."""

from __future__ import annotations

__all__ = ["DataFrameIngestor"]

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from grizz.ingestor import BaseIngestor as BaseDataFrameIngestor
from grizz.ingestor import setup_ingestor

from arkas.data.ingestor.base import BaseIngestor


class DataFrameIngestor(BaseIngestor):
    r"""Implement a DataFrame ingestor.

    Args:
        ingestor: The DataFrame ingestor or its configuration.
        out_key: The key of the DataFrame in the output dictionary.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.ingestor import Ingestor
    >>> from arkas.data.ingestor import DataFrameIngestor
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1, 2, 3, 4, 5],
    ...         "col2": ["1", "2", "3", "4", "5"],
    ...         "col3": ["a", "b", "c", "d", "e"],
    ...     }
    ... )
    >>> ingestor = DataFrameIngestor(ingestor=Ingestor(frame))
    >>> ingestor
    DataFrameIngestor(
      (ingestor): Ingestor(shape=(5, 3))
      (out_key): frame
    )
    >>> data = ingestor.ingest()
    >>> data
    {'frame': shape: (5, 3)
    ┌──────┬──────┬──────┐
    │ col1 ┆ col2 ┆ col3 │
    │ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ str  ┆ str  │
    ╞══════╪══════╪══════╡
    │ 1    ┆ 1    ┆ a    │
    │ 2    ┆ 2    ┆ b    │
    │ 3    ┆ 3    ┆ c    │
    │ 4    ┆ 4    ┆ d    │
    │ 5    ┆ 5    ┆ e    │
    └──────┴──────┴──────┘}

    ```
    """

    def __init__(self, ingestor: BaseDataFrameIngestor | dict, out_key: str = "frame") -> None:
        self._ingestor = setup_ingestor(ingestor)
        self._out_key = out_key

    def __repr__(self) -> str:
        args = repr_indent(
            repr_mapping(
                {
                    "ingestor": self._ingestor,
                    "out_key": self._out_key,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "ingestor": self._ingestor,
                    "out_key": self._out_key,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def ingest(self) -> dict:
        frame = self._ingestor.ingest()
        return {self._out_key: frame}
