r"""Contain the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator"]

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    r"""Define the base class to implement an evaluator.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator2 import AccuracyEvaluator
    >>> from arkas.state import AccuracyState
    >>> evaluator = AccuracyEvaluator(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> evaluator
    AccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )

    ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two evaluators are equal or not.

        Args:
            other: The other evaluator to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two evaluators are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.evaluator2 import AccuracyEvaluator
        >>> from arkas.state import AccuracyState
        >>> evaluator1 = AccuracyEvaluator(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> evaluator2 = AccuracyEvaluator(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> evaluator3 = AccuracyEvaluator(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 0, 0]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> evaluator1.equal(evaluator2)
        True
        >>> evaluator1.equal(evaluator3)
        False

        ```
        """
