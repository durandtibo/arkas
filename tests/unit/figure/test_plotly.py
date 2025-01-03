from __future__ import annotations

from unittest.mock import Mock

import pytest

from arkas.figure import PlotlyFigure, PlotlyFigureConfig
from arkas.testing import plotly_available
from arkas.utils.imports import is_plotly_available

if is_plotly_available():
    from plotly.graph_objects import Figure


@pytest.fixture
def figure() -> Figure:
    return Mock(spec=Figure)


##################################
#     Tests for PlotlyFigure     #
##################################


@plotly_available
def test_plotly_figure_repr(figure: Figure) -> None:
    assert repr(PlotlyFigure(figure)) == "PlotlyFigure(reactive=True)"


@plotly_available
def test_plotly_figure_str(figure: Figure) -> None:
    assert str(PlotlyFigure(figure)) == "PlotlyFigure(reactive=True)"


@plotly_available
def test_plotly_figure_close(figure: Figure) -> None:
    PlotlyFigure(figure).close()


@plotly_available
def test_plotly_figure_close_multiple(figure: Figure) -> None:
    fig = PlotlyFigure(figure)
    fig.close()
    fig.close()


@plotly_available
def test_plotly_figure_equal_true(figure: Figure) -> None:
    assert PlotlyFigure(figure).equal(PlotlyFigure(figure))


@plotly_available
def test_plotly_figure_equal_false_different_reactive(figure: Figure) -> None:
    assert not PlotlyFigure(figure).equal(PlotlyFigure(figure, reactive=False))


@plotly_available
def test_plotly_figure_equal_false_different_type(figure: Figure) -> None:
    assert not PlotlyFigure(figure).equal(42)


@plotly_available
@pytest.mark.parametrize("reactive", [True, False])
def test_plotly_figure_set_reactive(figure: Figure, reactive: bool) -> None:
    assert (
        PlotlyFigure(figure).set_reactive(reactive).equal(PlotlyFigure(figure, reactive=reactive))
    )


@plotly_available
@pytest.mark.parametrize("reactive", [True, False])
def test_plotly_figure_to_html(figure: Figure, reactive: bool) -> None:
    assert isinstance(PlotlyFigure(figure, reactive=reactive).to_html(), str)


########################################
#     Tests for PlotlyFigureConfig     #
########################################


@plotly_available
def test_plotly_figure_config_backend() -> None:
    assert PlotlyFigureConfig.backend() == "plotly"


@plotly_available
def test_plotly_figure_config_repr() -> None:
    assert repr(PlotlyFigureConfig()) == "PlotlyFigureConfig()"


@plotly_available
def test_plotly_figure_config_str() -> None:
    assert str(PlotlyFigureConfig()) == "PlotlyFigureConfig()"


@plotly_available
def test_plotly_figure_config_equal_true() -> None:
    assert PlotlyFigureConfig().equal(PlotlyFigureConfig())


@plotly_available
def test_plotly_figure_config_equal_false_different_kwargs() -> None:
    assert not PlotlyFigureConfig().equal(PlotlyFigureConfig(dpi=300))


@plotly_available
def test_plotly_figure_config_equal_false_different_type() -> None:
    assert not PlotlyFigureConfig().equal(42)


@plotly_available
def test_plotly_figure_config_get_args() -> None:
    assert PlotlyFigureConfig(dpi=300).get_args() == {"dpi": 300}
