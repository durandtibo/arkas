r"""Contain the implementation for matplotlib figures."""

from __future__ import annotations

__all__ = ["MatplotlibFigure", "MatplotlibFigureConfig"]

import base64
import io
import sys
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.figure.base import BaseFigure, BaseFigureConfig

if TYPE_CHECKING:
    from matplotlib.colors import Normalize

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )


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

    def set_reactive(self, reactive: bool) -> Self:
        return self.__class__(figure=self.figure, reactive=reactive)

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
        color_norm: The color normalization.
        **kwargs: Additional keyword arguments to pass to
            ``matplotlib.pyplot.subplots``.

    Example usage:

    ```pycon

    >>> from arkas.figure import MatplotlibFigureConfig
    >>> config = MatplotlibFigureConfig()
    >>> config
    MatplotlibFigureConfig(color_norm=None)

    ```
    """

    def __init__(self, color_norm: Normalize | None = None, **kwargs: Any) -> None:
        self._color_norm = color_norm
        self._kwargs = kwargs

    def __repr__(self) -> str:
        args = repr_mapping_line({"color_norm": self._color_norm} | self.get_args())
        return f"{self.__class__.__qualname__}({args})"

    @classmethod
    def backend(cls) -> str:
        return "matplotlib"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        # color_norm is excluded from the comparison as it is not straightforward
        # to compare to Normalize objects.
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def get_args(self) -> dict:
        return self._kwargs

    def get_color_norm(self) -> Normalize | None:
        r"""Get the color normalization.

        Returns:
            The color normalization.

        Example usage:

        ```pycon

        >>> from matplotlib.colors import LogNorm
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> config = MatplotlibFigureConfig(color_norm=LogNorm())
        >>> config.get_color_norm()
        <matplotlib.colors.LogNorm object at 0x...>

        ```
        """
        return self._color_norm
