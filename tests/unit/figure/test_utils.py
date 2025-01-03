from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from arkas.figure import MatplotlibFigure, figure2html


@pytest.fixture
def figure() -> plt.Figure:
    return plt.subplots()[0]


#################################
#     Tests for figure2html     #
#################################


def test_figure2html(figure: plt.Figure) -> None:
    assert isinstance(figure2html(MatplotlibFigure(figure)), str)


@pytest.mark.parametrize("close_fig", [True, False])
def test_figure2html_close_fig(close_fig: bool, figure: plt.Figure) -> None:
    assert isinstance(figure2html(MatplotlibFigure(figure), close_fig=close_fig), str)


@pytest.mark.parametrize("reactive", [True, False])
def test_figure2html_reactive(reactive: bool, figure: plt.Figure) -> None:
    assert isinstance(figure2html(MatplotlibFigure(figure), reactive=reactive), str)
