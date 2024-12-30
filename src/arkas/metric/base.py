r"""Contain the base class to implement a metric."""

from __future__ import annotations

__all__ = ["BaseMetric"]

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    r"""Define the base class to implement a metric.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import AccuracyMetric
    >>> from arkas.state import AccuracyState
    >>> metric = AccuracyMetric(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> metric
    AccuracyMetric(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )

    ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two metrics are equal or not.

        Args:
            other: The other metric to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two metrics are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.metric import AccuracyMetric
        >>> from arkas.state import AccuracyState
        >>> metric1 = AccuracyMetric(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> metric2 = AccuracyMetric(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> metric3 = AccuracyMetric(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 0, 0]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> metric1.equal(metric2)
        True
        >>> metric1.equal(metric3)
        False

        ```
        """
