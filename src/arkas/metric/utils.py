r"""Contain utility functions to compute metrics."""

from __future__ import annotations

__all__ = ["multi_isnan", "preprocess_true_pred"]


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


def preprocess_true_pred(
    y_true: np.ndarray, y_pred: np.ndarray, nan: str = "keep"
) -> tuple[np.ndarray, np.ndarray]:
    r"""Preprocess ``y_true`` and ``y_pred`` arrays.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        nan: Indicate how to process the nan values.
            If ``'keep'``, the nan values are kept.
            If ``'remove'``, the nan values are removed.

    Returns:
        A tuple with the preprocessed ``y_true`` and ``y_pred``
            arrays.

    Raises:
        RuntimeError: if an invalid value is passed to ``nan``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric.utils import preprocess_true_pred
    >>> y_true = np.array([1, 0, 0, 1, 1, float("nan")])
    >>> y_pred = np.array([0, 1, 0, 1, float("nan"), 1])
    >>> preprocess_true_pred(y_true, y_pred)
    (array([ 1.,  0.,  0.,  1.,  1., nan]), array([ 0.,  1.,  0.,  1., nan,  1.]))
    >>> preprocess_true_pred(y_true, y_pred, nan="remove")
    (array([1., 0., 0., 1.]), array([0., 1., 0., 1.]))

    ```
    """
    if nan == "keep":
        return y_true, y_pred
    if nan == "remove":
        mask = np.logical_not(multi_isnan([y_true, y_pred]))
        return y_true[mask], y_pred[mask]
    msg = f"Incorrect 'nan': {nan}. The valid values are 'keep' and 'remove'"
    raise RuntimeError(msg)
