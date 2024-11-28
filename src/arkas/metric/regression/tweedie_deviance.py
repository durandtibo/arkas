r"""Implement the mean squared error metrics."""

from __future__ import annotations

__all__ = ["mean_tweedie_deviance"]

from typing import TYPE_CHECKING

from sklearn import metrics

from arkas.metric.utils import preprocess_pred

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


def mean_tweedie_deviance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    powers: Sequence[float] = (0,),
    prefix: str = "",
    suffix: str = "",
    drop_nan: bool = False,
) -> dict[str, float]:
    r"""Return the mean squared error (MSE).

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        powers: The Tweedie power parameter. The higher power the less
            weight is given to extreme deviations between true and
            predicted targets.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        drop_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import mean_tweedie_deviance
    >>> mean_tweedie_deviance(
    ...     y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
    ... )
    {'count': 5, 'mean_tweedie_deviance_power_0': 0.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=drop_nan
    )

    count = y_true.size
    out = {f"{prefix}count{suffix}": count}
    for power in powers:
        score = float("nan")
        if count > 0:
            score = metrics.mean_tweedie_deviance(y_true=y_true, y_pred=y_pred, power=power)
        out[f"{prefix}mean_tweedie_deviance_power_{power}{suffix}"] = float(score)
    return out
