from __future__ import annotations

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import PlotColumnPlotter, Plotter
from arkas.plotter.plot_column import MatplotlibFigureCreator
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


#######################################
#     Tests for PlotColumnPlotter     #
#######################################


def test_plot_column_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert repr(PlotColumnPlotter(DataFrameState(dataframe))).startswith("PlotColumnPlotter(")


def test_plot_column_plotter_str(dataframe: pl.DataFrame) -> None:
    assert str(PlotColumnPlotter(DataFrameState(dataframe))).startswith("PlotColumnPlotter(")


def test_plot_column_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotColumnPlotter(DataFrameState(dataframe)).compute(), Plotter)


def test_plot_column_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert PlotColumnPlotter(DataFrameState(dataframe)).equal(
        PlotColumnPlotter(DataFrameState(dataframe))
    )


def test_plot_column_plotter_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not PlotColumnPlotter(DataFrameState(dataframe)).equal(
        PlotColumnPlotter(DataFrameState(pl.DataFrame()))
    )


def test_plot_column_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not PlotColumnPlotter(DataFrameState(dataframe)).equal(42)


def test_plot_column_plotter_equal_nan_true() -> None:
    assert PlotColumnPlotter(
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
        PlotColumnPlotter(
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


def test_plot_column_plotter_equal_nan_false() -> None:
    assert not PlotColumnPlotter(
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
        PlotColumnPlotter(
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


def test_plot_column_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = PlotColumnPlotter(DataFrameState(dataframe)).plot()
    assert len(figures) == 1
    assert isinstance(figures["plot_column"], MatplotlibFigure)


def test_plot_column_plotter_plot_empty() -> None:
    figures = PlotColumnPlotter(DataFrameState(pl.DataFrame())).plot()
    assert len(figures) == 1
    assert figures["plot_column"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_plot_column_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = PlotColumnPlotter(DataFrameState(dataframe)).plot(prefix="prefix_", suffix="_suffix")
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
        MatplotlibFigureCreator().create(DataFrameState(dataframe)),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            DataFrameState(
                dataframe,
                figure_config=MatplotlibFigureConfig(yscale="symlog", init={"figsize": (3, 3)}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(DataFrameState(pl.DataFrame()))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
