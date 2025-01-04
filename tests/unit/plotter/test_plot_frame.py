from __future__ import annotations

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, get_default_config
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import PlotDataFramePlotter, Plotter
from arkas.plotter.plot_frame import MatplotlibFigureCreator
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


##########################################
#     Tests for PlotDataFramePlotter     #
##########################################


def test_plot_dataframe_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert repr(PlotDataFramePlotter(DataFrameState(dataframe))).startswith("PlotDataFramePlotter(")


def test_plot_dataframe_plotter_str(dataframe: pl.DataFrame) -> None:
    assert str(PlotDataFramePlotter(DataFrameState(dataframe))).startswith("PlotDataFramePlotter(")


def test_plot_dataframe_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotDataFramePlotter(DataFrameState(dataframe)).compute(), Plotter)


def test_plot_dataframe_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert PlotDataFramePlotter(DataFrameState(dataframe)).equal(
        PlotDataFramePlotter(DataFrameState(dataframe))
    )


def test_plot_dataframe_plotter_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not PlotDataFramePlotter(DataFrameState(dataframe)).equal(
        PlotDataFramePlotter(DataFrameState(pl.DataFrame()))
    )


def test_plot_dataframe_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not PlotDataFramePlotter(DataFrameState(dataframe)).equal(42)


def test_plot_dataframe_plotter_equal_nan_true() -> None:
    assert PlotDataFramePlotter(
        DataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            )
        )
    ).equal(
        PlotDataFramePlotter(
            DataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                        "col3": [0, 0, 0, 0, 1, 1, 1],
                    }
                )
            )
        ),
        equal_nan=True,
    )


def test_plot_dataframe_plotter_equal_nan_false() -> None:
    assert not PlotDataFramePlotter(
        DataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            )
        )
    ).equal(
        PlotDataFramePlotter(
            DataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                        "col3": [0, 0, 0, 0, 1, 1, 1],
                    }
                )
            )
        )
    )


def test_plot_dataframe_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = PlotDataFramePlotter(DataFrameState(dataframe)).plot()
    assert len(figures) == 1
    assert isinstance(figures["plot_column"], MatplotlibFigure)


def test_plot_dataframe_plotter_plot_empty() -> None:
    figures = PlotDataFramePlotter(DataFrameState(pl.DataFrame())).plot()
    assert len(figures) == 1
    assert figures["plot_column"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_plot_dataframe_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = PlotDataFramePlotter(DataFrameState(dataframe)).plot(
        prefix="prefix_", suffix="_suffix"
    )
    assert len(figures) == 1
    assert isinstance(figures["prefix_plot_column_suffix"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(frame=dataframe, config=get_default_config()),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(frame=pl.DataFrame(), config=get_default_config())
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
