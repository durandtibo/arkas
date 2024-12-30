r"""Contain the base class to implement a plotter."""

from __future__ import annotations

__all__ = ["BasePlotter"]

from abc import ABC, abstractmethod
from typing import Any


class BasePlotter(ABC):
    r"""Define the base class to implement a plotter.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.plotter import Plotter
    >>> plotter = Plotter()
    >>> plotter
    Plotter(count=0)

    ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two plotters are equal or not.

        Args:
            other: The other plotter to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two plotters are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from arkas.plotter import Plotter
        >>> plotter1 = Plotter()
        >>> plotter2 = Plotter()
        >>> plotter3 = Plotter({"fig": None})
        >>> plotter1.equal(plotter2)
        True
        >>> plotter1.equal(plotter3)
        False

        ```
        """
