r"""Contain utility functions to compute metrics."""

from __future__ import annotations

__all__ = ["multi_isnan"]


from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def multi_isnan(arrays: Sequence[np.ndarray]) -> np.ndarray:
    r"""Test element-wise for NaN for all input arrays and return result
    as a boolean array.

    Args:
        arrays: The input arrays to test. All the arrays must have the
            same shape.

    Returns:
        A boolean array. ``True`` where any array is NaN,
            ``False`` otherwise.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric.utils import multi_isnan
    >>> mask = multi_isnan(
    ...     [np.array([1, 0, 0, 1, float("nan")]), np.array([1, float("nan"), 0, 1, 1])]
    ... )
    >>> mask
    array([False,  True, False, False,  True])

    ```
    """
    if len(arrays) == 0:
        msg = "'arrays' cannot be empty"
        raise RuntimeError(msg)
    mask = np.isnan(arrays[0])
    for arr in arrays[1:]:
        mask = np.logical_or(mask, np.isnan(arr))
    return mask
