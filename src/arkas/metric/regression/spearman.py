r"""Implement the Spearman correlation coefficient and p-value for
testing non-correlation."""

from __future__ import annotations

__all__ = ["spearman_correlation"]


from typing import TYPE_CHECKING

from arkas.metric.utils import preprocess_pred
from arkas.utils.imports import check_scipy, is_scipy_available

if is_scipy_available():
    from scipy.stats import spearmanr

if TYPE_CHECKING:
    import numpy as np


def spearman_correlation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float]:
    r"""Return the Spearman correlation coefficient and p-value for
    testing non-correlation.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import spearman_correlation
    >>> spearman_correlation(
    ...     y_true=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ...     y_pred=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ... )
    {'count': 9, 'spearman_coeff': 1.0, 'spearman_pvalue': 0.0}

    ```
    """
    check_scipy()
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=ignore_nan
    )

    count = y_true.size
    coeff, pvalue = float("nan"), float("nan")
    if count > 0:
        result = spearmanr(y_true, y_pred)
        coeff, pvalue = float(result.statistic), float(result.pvalue)
    return {
        f"{prefix}count{suffix}": count,
        f"{prefix}spearman_coeff{suffix}": coeff,
        f"{prefix}spearman_pvalue{suffix}": pvalue,
    }
