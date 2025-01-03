from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from arkas.figure import MatplotlibFigure, MatplotlibFigureConfig


@pytest.fixture
def figure() -> plt.Figure:
    return plt.subplots()[0]


######################################
#     Tests for MatplotlibFigure     #
######################################


def test_matplotlib_figure_repr(figure: plt.Figure) -> None:
    assert repr(MatplotlibFigure(figure)) == "MatplotlibFigure(reactive=True)"


def test_matplotlib_figure_str(figure: plt.Figure) -> None:
    assert str(MatplotlibFigure(figure)) == "MatplotlibFigure(reactive=True)"


def test_matplotlib_figure_close(figure: plt.Figure) -> None:
    MatplotlibFigure(figure).close()


def test_matplotlib_figure_close_multiple(figure: plt.Figure) -> None:
    fig = MatplotlibFigure(figure)
    fig.close()
    fig.close()


def test_matplotlib_figure_equal_true(figure: plt.Figure) -> None:
    assert MatplotlibFigure(figure).equal(MatplotlibFigure(figure))


def test_matplotlib_figure_equal_false_different_reactive(figure: plt.Figure) -> None:
    assert not MatplotlibFigure(figure).equal(MatplotlibFigure(figure, reactive=False))


def test_matplotlib_figure_equal_false_different_type(figure: plt.Figure) -> None:
    assert not MatplotlibFigure(figure).equal(42)


@pytest.mark.parametrize("reactive", [True, False])
def test_matplotlib_figure_to_html(figure: plt.Figure, reactive: bool) -> None:
    assert isinstance(MatplotlibFigure(figure, reactive=reactive).to_html(), str)


############################################
#     Tests for MatplotlibFigureConfig     #
############################################


def test_matplotlib_figure_config_backend() -> None:
    assert MatplotlibFigureConfig.backend() == "matplotlib"


def test_matplotlib_figure_config_repr() -> None:
    assert repr(MatplotlibFigureConfig()) == "MatplotlibFigureConfig()"


def test_matplotlib_figure_config_str() -> None:
    assert str(MatplotlibFigureConfig()) == "MatplotlibFigureConfig()"


def test_matplotlib_figure_config_equal_true() -> None:
    assert MatplotlibFigureConfig().equal(MatplotlibFigureConfig())


def test_matplotlib_figure_config_equal_false_different_kwargs() -> None:
    assert not MatplotlibFigureConfig().equal(MatplotlibFigureConfig(dpi=300))


def test_matplotlib_figure_config_equal_false_different_type() -> None:
    assert not MatplotlibFigureConfig().equal(42)


def test_matplotlib_figure_config_get_args() -> None:
    assert MatplotlibFigureConfig(dpi=300).get_args() == {"dpi": 300}