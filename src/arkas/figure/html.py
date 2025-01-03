r"""Contain the implementation for matplotlib figures."""

from __future__ import annotations

__all__ = ["HtmlFigure"]

from typing import Any

from arkas.figure.base import BaseFigure


class HtmlFigure(BaseFigure):
    r"""Implement a simple HTML figure.

    Args:
        figure: The HTML code of the figure.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from arkas.figure import HtmlFigure
    >>> fig = HtmlFigure()
    >>> fig
    HtmlFigure()

    ```
    """

    def __init__(self, figure: str = "") -> None:
        self._figure = figure

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def close(self) -> None:
        pass

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        if not isinstance(other, self.__class__):
            return False
        return self._figure == other._figure

    def to_html(self) -> str:
        return self._figure