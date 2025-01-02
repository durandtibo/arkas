"""Contain the base class to implement a figure."""

from __future__ import annotations

__all__ = ["BaseFigure", "BaseFigureConfig"]

from abc import ABC, abstractmethod
from typing import Any


class BaseFigure(ABC):
    r"""Define the base class to implement a figure."""

    @abstractmethod
    def close(self) -> None:
        r"""Close the figure."""

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two figures are equal or not.

        Args:
            other: The other object to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.
        """

    @abstractmethod
    def to_html(self) -> str:
        r"""Export the figure to a HTML code.

        Returns:
            The HTML code of the figure.
        """


class BaseFigureConfig(ABC):
    r"""Define the base class to implement a figure config."""

    @property
    @abstractmethod
    def backend(self) -> str:
        r"""The backend to generate the figure."""

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two configs are equal or not.

        Args:
            other: The other object to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.
        """

    @abstractmethod
    def get_args(self) -> dict:
        r"""Get the config arguments.

        Returns:
            The config arguments.
        """
