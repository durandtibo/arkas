r"""Implement the Pearson correlation metrics."""

from __future__ import annotations

__all__ = ["pearsonr"]


from typing import TYPE_CHECKING

from arkas.metric.utils import preprocess_pred
from arkas.utils.imports import check_scipy, is_scipy_available

if is_scipy_available():
    from scipy import stats

if TYPE_CHECKING:
    import numpy as np


def pearsonr(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    alternative: str = "two-sided",
    prefix: str = "",
    suffix: str = "",
    ignore_nan: bool = False,
) -> dict[str, float]:
    r"""Return the Pearson correlation coefficient and p-value for
    testing non-correlation.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        alternative: The alternative hypothesis. Default is 'two-sided'.
            The following options are available:
            - 'two-sided': the correlation is nonzero
            - 'less': the correlation is negative (less than zero)
            - 'greater': the correlation is positive (greater than zero)
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        ignore_nan: If ``True``, the NaN values are ignored while
            computing the metrics, otherwise an exception is raised.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import pearsonr
    >>> pearsonr(
    ...     y_true=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ...     y_pred=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ... )
    {'count': 5, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0}

    ```
    """
    check_scipy()
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), remove_nan=ignore_nan
    )

    count = y_true.size
    coeff, pvalue = float("nan"), float("nan")
    if count > 0:
        result = stats.pearsonr(x=y_true, y=y_pred, alternative=alternative)
        coeff, pvalue = float(result.statistic), float(result.pvalue)
    return {
        f"{prefix}count{suffix}": count,
        f"{prefix}pearson_coeff{suffix}": coeff,
        f"{prefix}pearson_pvalue{suffix}": pvalue,
    }
