from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from arkas.figure import MatplotlibFigure, figure2html

#################################
#     Tests for figure2html     #
#################################


def test_figure2html() -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(MatplotlibFigure(fig)), str)


@pytest.mark.parametrize("close_fig", [True, False])
def test_figure2html_close_fig(close_fig: bool) -> None:
    fig, _ = plt.subplots()
    assert isinstance(figure2html(MatplotlibFigure(fig), close_fig=close_fig), str)
