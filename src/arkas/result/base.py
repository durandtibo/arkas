r"""Contain the base class to implement a section."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod
from typing import Any


class BaseResult(ABC):
    r"""Define the base class to manage results.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AccuracyResult
    >>> result = AccuracyResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    AccuracyResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """

    @abstractmethod
    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict:
        r"""Return the metrics associated to the result.

        Args:
            prefix: The key prefix in the returned dictionary.
            suffix: The key suffix in the returned dictionary.

        Returns:
            The metrics.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.result import AccuracyResult
        >>> result = AccuracyResult(
        ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ... )
        >>> result.compute_metrics()
        {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

        ```
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two results are equal or not.

        Args:
            other: The other result to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two results are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.result import AccuracyResult
        >>> res1 = AccuracyResult(
        ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ... )
        >>> res2 = AccuracyResult(
        ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ... )
        >>> res3 = AccuracyResult(
        ...     y_true=np.array([1, 0, 0, 0, 0]), y_pred=np.array([1, 0, 0, 1, 1])
        ... )
        >>> res1.equal(res2)
        True
        >>> res1.equal(res3)
        False

        ```
        """

    @abstractmethod
    def generate_figures(self, prefix: str = "", suffix: str = "") -> dict:
        r"""Return the figures associated to the result.

        Args:
            prefix: The key prefix in the returned dictionary.
            suffix: The key suffix in the returned dictionary.

        Returns:
            The figures.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.result import AccuracyResult
        >>> result = AccuracyResult(
        ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ... )
        >>> result.generate_figures()
        {}

        ```
        """
