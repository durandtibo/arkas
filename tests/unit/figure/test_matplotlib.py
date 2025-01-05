from __future__ import annotations

import pytest
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

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
def test_matplotlib_figure_set_reactive(figure: plt.Figure, reactive: bool) -> None:
    assert (
        MatplotlibFigure(figure)
        .set_reactive(reactive)
        .equal(MatplotlibFigure(figure, reactive=reactive))
    )


@pytest.mark.parametrize("reactive", [True, False])
def test_matplotlib_figure_to_html(figure: plt.Figure, reactive: bool) -> None:
    assert isinstance(MatplotlibFigure(figure, reactive=reactive).to_html(), str)


############################################
#     Tests for MatplotlibFigureConfig     #
############################################


def test_matplotlib_figure_config_repr() -> None:
    assert repr(MatplotlibFigureConfig()) == "MatplotlibFigureConfig(color_norm=None)"


def test_matplotlib_figure_config_str() -> None:
    assert str(MatplotlibFigureConfig()) == "MatplotlibFigureConfig(color_norm=None)"


def test_matplotlib_figure_config_backend() -> None:
    assert MatplotlibFigureConfig.backend() == "matplotlib"


def test_matplotlib_figure_config_clone() -> None:
    config = MatplotlibFigureConfig(color_norm=LogNorm(), dpi=300)
    cloned_config = config.clone()
    assert config.equal(cloned_config)
    assert isinstance(config.get_color_norm(), LogNorm)
    assert isinstance(cloned_config.get_color_norm(), LogNorm)


def test_matplotlib_figure_config_equal_true() -> None:
    assert MatplotlibFigureConfig().equal(MatplotlibFigureConfig())


def test_matplotlib_figure_config_equal_false_different_kwargs() -> None:
    assert not MatplotlibFigureConfig().equal(MatplotlibFigureConfig(dpi=300))


def test_matplotlib_figure_config_equal_false_different_type() -> None:
    assert not MatplotlibFigureConfig().equal(42)


def test_matplotlib_figure_config_get_init_args() -> None:
    assert MatplotlibFigureConfig(dpi=300).get_init_args() == {"dpi": 300}


def test_matplotlib_figure_config_get_color_norm() -> None:
    color_norm = LogNorm()
    color_norm2 = MatplotlibFigureConfig(color_norm=color_norm).get_color_norm()
    assert color_norm is not color_norm2
    assert isinstance(color_norm2, LogNorm)


def test_matplotlib_figure_config_get_color_norm_default() -> None:
    assert MatplotlibFigureConfig().get_color_norm() is None
