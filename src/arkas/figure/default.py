r"""Contain the default figure config."""

from __future__ import annotations

__all__ = ["DefaultFigureConfig"]

from typing import Any

from arkas.figure.base import BaseFigureConfig


class DefaultFigureConfig(BaseFigureConfig):
    r"""Implement the default figure config.

    Example usage:

    ```pycon

    >>> from arkas.figure import DefaultFigureConfig
    >>> config = DefaultFigureConfig()
    >>> config
    DefaultFigureConfig()

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    @classmethod
    def backend(cls) -> str:
        return "default"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return isinstance(other, self.__class__)

    def get_args(self) -> dict:
        return {}
