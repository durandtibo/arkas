r"""Implement some utility functions for ``numpy.ndarray``s."""

from __future__ import annotations

__all__ = ["to_array"]

from typing import TYPE_CHECKING, Any

import polars as pl
from coola.utils.array import to_array as coola_to_array

if TYPE_CHECKING:
    import numpy as np


def to_array(data: Any) -> np.ndarray:
    r"""Convert the input to a ``numpy.ndarray``.

    Args:
        data: The data to convert to a NumPy array.

    Returns:
        A NumPy array.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.utils.array import to_array
    >>> to_array([1, 2, 3, 4, 5])
    array([1, 2, 3, 4, 5])
    >>> to_array(pl.Series([1, 2, 3, 4, 5]))
    array([1, 2, 3, 4, 5])
    >>> to_array(pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [0, 1, 0, 1, 0]}))
    array([[1, 0], [2, 1], [3, 0], [4, 1], [5, 0]])

    ```
    """
    if isinstance(data, pl.Series):
        return data.to_numpy()
    if isinstance(data, pl.DataFrame):
        return data.to_numpy()
    return coola_to_array(data)
