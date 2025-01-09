from __future__ import annotations

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import ContinuousSeriesPlotter, Plotter
from arkas.plotter.continuous_series import MatplotlibFigureCreator
from arkas.state import SeriesState


@pytest.fixture
def series() -> pl.Series:
    return pl.Series("col1", [1, 2, 3, 4, 5, 6, 7])


#############################################
#     Tests for ContinuousSeriesPlotter     #
#############################################


def test_continuous_series_plotter_repr(series: pl.Series) -> None:
    assert repr(ContinuousSeriesPlotter(SeriesState(series))).startswith("ContinuousSeriesPlotter(")


def test_continuous_series_plotter_str(series: pl.Series) -> None:
    assert str(ContinuousSeriesPlotter(SeriesState(series))).startswith("ContinuousSeriesPlotter(")


def test_continuous_series_plotter_compute(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesPlotter(SeriesState(series)).compute(), Plotter)


def test_continuous_series_plotter_equal_true(series: pl.Series) -> None:
    assert ContinuousSeriesPlotter(SeriesState(series)).equal(
        ContinuousSeriesPlotter(SeriesState(series))
    )


def test_continuous_series_plotter_equal_false_different_state(series: pl.Series) -> None:
    assert not ContinuousSeriesPlotter(SeriesState(series)).equal(
        ContinuousSeriesPlotter(SeriesState(pl.Series([])))
    )


def test_continuous_series_plotter_equal_false_different_type(series: pl.Series) -> None:
    assert not ContinuousSeriesPlotter(SeriesState(series)).equal(42)


def test_continuous_series_plotter_equal_nan_true() -> None:
    assert ContinuousSeriesPlotter(
        SeriesState(pl.Series("col1", [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")]))
    ).equal(
        ContinuousSeriesPlotter(
            SeriesState(pl.Series("col1", [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")]))
        ),
        equal_nan=True,
    )


def test_continuous_series_plotter_equal_nan_false() -> None:
    assert not ContinuousSeriesPlotter(
        SeriesState(pl.Series("col1", [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")]))
    ).equal(
        ContinuousSeriesPlotter(
            SeriesState(pl.Series("col1", [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")]))
        )
    )


def test_continuous_series_plotter_plot(series: pl.Series) -> None:
    figures = ContinuousSeriesPlotter(SeriesState(series)).plot()
    assert len(figures) == 1
    assert isinstance(figures["continuous_histogram"], MatplotlibFigure)


def test_continuous_series_plotter_plot_empty() -> None:
    figures = ContinuousSeriesPlotter(SeriesState(pl.Series())).plot()
    assert len(figures) == 1
    assert figures["continuous_histogram"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_continuous_series_plotter_plot_prefix_suffix(series: pl.Series) -> None:
    figures = ContinuousSeriesPlotter(SeriesState(series)).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_continuous_histogram_suffix"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create(series: pl.Series) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(SeriesState(series)),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_1_row() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(SeriesState(pl.Series("col1", [0]))),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config(series: pl.Series) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            SeriesState(
                series,
                figure_config=MatplotlibFigureConfig(yscale="symlog", init={}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(SeriesState(pl.Series()))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
