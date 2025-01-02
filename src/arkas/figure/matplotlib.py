r"""Contain the implementation for matplotlib figures."""

from __future__ import annotations

__all__ = ["MatplotlibFigureConfig"]

from typing import Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.figure.base import BaseFigureConfig


class MatplotlibFigureConfig(BaseFigureConfig):
    r"""Implement the matplotlib figure config.

    Args:
        **kwargs: Additional keyword arguments to pass to matplotlib
            functions. The valid arguments depend on the context.
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line(self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    @classmethod
    def backend(cls) -> str:
        return "matplotlib"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def get_args(self) -> dict:
        return self._kwargs
