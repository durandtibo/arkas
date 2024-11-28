r"""Implement the regression error metrics."""

from __future__ import annotations

__all__ = ["regression_errors"]


from typing import TYPE_CHECKING

from arkas.metric.regression.abs_error import mean_absolute_error, median_absolute_error
from arkas.metric.regression.mse import mean_squared_error
from arkas.metric.utils import preprocess_pred

if TYPE_CHECKING:
    import numpy as np


def regression_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    drop_nan: bool = False,
) -> dict[str, float]:
    r"""Return the regression error metrics.

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
    >>> from arkas.metric import regression_errors
    >>> regression_errors(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]))
    {'count': 5,
     'mean_absolute_error': 0.0,
     'median_absolute_error': 0.0,
     'mean_squared_error': 0.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=drop_nan
    )

    return (
        mean_absolute_error(y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix)
        | median_absolute_error(y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix)
        | mean_squared_error(y_true=y_true, y_pred=y_pred, prefix=prefix, suffix=suffix)
    )
