r"""Contain the base class to implement a section."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod


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
    {'accuracy': 1.0, 'count': 5}

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
        {'accuracy': 1.0, 'count': 5}

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
