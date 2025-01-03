r"""Contain figures."""

from __future__ import annotations

__all__ = [
    "BaseFigure",
    "BaseFigureConfig",
    "DefaultFigureConfig",
    "HtmlFigure",
    "MatplotlibFigure",
    "MatplotlibFigureConfig",
]

from arkas.figure.base import BaseFigure, BaseFigureConfig
from arkas.figure.default import DefaultFigureConfig
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
