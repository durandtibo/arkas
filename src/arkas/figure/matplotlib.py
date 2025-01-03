r"""Contain the implementation for matplotlib figures."""

from __future__ import annotations

__all__ = ["MatplotlibFigure", "MatplotlibFigureConfig"]

import base64
import io
from typing import Any

import matplotlib.pyplot as plt
from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.figure.base import BaseFigure, BaseFigureConfig


class MatplotlibFigure(BaseFigure):
    r"""Implement the matplotlib figure.

    Args:
        figure: The matplotlib figure.
        reactive: If ``True``, the generated is configured to be
            reactive to the screen size.

    Example usage:

    ```pycon

    >>> from matplotlib import pyplot as plt
    >>> from arkas.figure import MatplotlibFigure
    >>> fig = MatplotlibFigure(plt.subplots()[0])
    >>> fig
    MatplotlibFigure(reactive=True)

    ```
    """

    def __init__(self, figure: plt.Figure, reactive: bool = True) -> None:
        self._figure = figure
        self._reactive = reactive

    def __repr__(self) -> str:
        args = repr_mapping_line({"reactive": self._reactive})
        return f"{self.__class__.__qualname__}({args})"

    @property
    def figure(self) -> plt.Figure:
        return self._figure

    def close(self) -> None:
        plt.close(self._figure)

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.figure, other.figure, equal_nan=equal_nan)
            and self._reactive == other._reactive
        )

    def to_html(self) -> str:
        self._figure.tight_layout()
        img = io.BytesIO()
        self._figure.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        data = base64.b64encode(img.getvalue()).decode("utf-8")
        style = 'style="width:100%; height:auto;" ' if self._reactive else ""
        return f'<img {style}src="data:image/png;charset=utf-8;base64, {data}">'


class MatplotlibFigureConfig(BaseFigureConfig):
    r"""Implement the matplotlib figure config.

    Args:
        **kwargs: Additional keyword arguments to pass to matplotlib
            functions. The valid arguments depend on the context.

    Example usage:

    ```pycon

    >>> from arkas.figure import MatplotlibFigureConfig
    >>> config = MatplotlibFigureConfig(dpi=300)
    >>> config
    MatplotlibFigureConfig(dpi=300)

    ```
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
