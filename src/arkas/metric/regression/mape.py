r"""Implement the mean absolute percentage error (MAPE) metrics."""

from __future__ import annotations

__all__ = ["mean_absolute_percentage_error"]


from typing import TYPE_CHECKING

from sklearn import metrics

from arkas.metric.utils import preprocess_pred

if TYPE_CHECKING:
    import numpy as np


def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    drop_nan: bool = False,
) -> dict[str, float]:
    r"""Return the mean absolute percentage error (MAPE).

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        drop_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import mean_absolute_percentage_error
    >>> mean_absolute_percentage_error(
    ...     y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
    ... )
    {'count': 5, 'mean_absolute_percentage_error': 0.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=drop_nan
    )

    count = y_true.size
    error = float("nan")
    if count > 0:
        error = float(metrics.mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
    return {
        f"{prefix}count{suffix}": count,
        f"{prefix}mean_absolute_percentage_error{suffix}": error,
    }
