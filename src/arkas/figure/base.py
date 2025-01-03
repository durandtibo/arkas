"""Contain the base class to implement a figure."""

from __future__ import annotations

__all__ = ["BaseFigure", "BaseFigureConfig"]

from abc import ABC, abstractmethod
from typing import Any


class BaseFigure(ABC):
    r"""Define the base class to implement a figure.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from arkas.figure import MatplotlibFigure
    >>> fig = MatplotlibFigure(plt.subplots()[0])
    >>> fig
    MatplotlibFigure(reactive=True)

    ```
    """

    @abstractmethod
    def close(self) -> None:
        r"""Close the figure.

        Example usage:

        ```pycon

        >>> from matplotlib import pyplot as plt
        >>> from arkas.figure import MatplotlibFigure
        >>> fig = MatplotlibFigure(plt.subplots()[0])
        >>> fig.close()

        ```
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two figures are equal or not.

        Args:
            other: The other object to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from matplotlib import pyplot as plt
        >>> from arkas.figure import MatplotlibFigure
        >>> fig = plt.subplots()[0]
        >>> fig1 = MatplotlibFigure(fig)
        >>> fig2 = MatplotlibFigure(fig)
        >>> fig3 = MatplotlibFigure(fig, reactive=False)
        >>> fig1.equal(fig2)
        True
        >>> fig1.equal(fig3)
        False

        ```
        """

    @abstractmethod
    def to_html(self) -> str:
        r"""Export the figure to a HTML code.

        Returns:
            The HTML code of the figure.

        ```pycon

        >>> from matplotlib import pyplot as plt
        >>> from arkas.figure import MatplotlibFigure
        >>> fig = MatplotlibFigure(plt.subplots()[0])
        >>> html = fig.to_html()

        ```
        """


class BaseFigureConfig(ABC):
    r"""Define the base class to implement a figure config.

    Example usage:

    ```pycon

    >>> from arkas.figure import MatplotlibFigureConfig
    >>> config = MatplotlibFigureConfig(dpi=300)
    >>> config
    MatplotlibFigureConfig(dpi=300)

    ```
    """

    @classmethod
    @abstractmethod
    def backend(cls) -> str:
        r"""Return the backend to generate the figure.

        Example usage:

        ```pycon

        >>> from arkas.figure import MatplotlibFigureConfig
        >>> backend = MatplotlibFigureConfig.backend()
        >>> backend
        matplotlib

        ```
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two configs are equal or not.

        Args:
            other: The other object to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from arkas.figure import MatplotlibFigureConfig
        >>> config1 = MatplotlibFigureConfig(dpi=300)
        >>> config2 = MatplotlibFigureConfig(dpi=300)
        >>> config3 = MatplotlibFigureConfig()
        >>> config1.equal(config2)
        True
        >>> config1.equal(config3)
        False

        ```
        """

    @abstractmethod
    def get_args(self) -> dict:
        r"""Get the config arguments.

        Returns:
            The config arguments.

        Example usage:

        ```pycon

        >>> from arkas.figure import MatplotlibFigureConfig
        >>> config = MatplotlibFigureConfig(dpi=300)
        >>> config.get_args()
        {'dpi': 300}

        ```
        """