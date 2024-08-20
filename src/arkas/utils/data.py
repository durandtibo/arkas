r"""Contain data utility functions."""

from __future__ import annotations

__all__ = ["find_keys"]


from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping


def find_keys(data: Mapping | pl.DataFrame) -> set:
    r"""Find all the keys in the input data.

    Args:
        data: The input data.

    Returns:
        The set of keys.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.utils.data import find_keys
    >>> keys = find_keys(
    ...     {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}
    ... )
    >>> sorted(keys)
    ['pred', 'target']
    >>> keys = find_keys(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [3, 2, 0, 1, 0]}))
    >>> sorted(keys)
    ['pred', 'target']

    ```
    """
    if isinstance(data, pl.DataFrame):
        return set(data.columns)
    return set(data.keys())
