from __future__ import annotations

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import HexbinColumnPlotter, Plotter
from arkas.plotter.hexbin_column import MatplotlibFigureCreator
from arkas.state import ScatterDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


#########################################
#     Tests for HexbinColumnPlotter     #
#########################################


def test_hexbin_column_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("HexbinColumnPlotter(")


def test_hexbin_column_plotter_str(dataframe: pl.DataFrame) -> None:
    assert str(
        HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("HexbinColumnPlotter(")


def test_hexbin_column_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).compute(),
        Plotter,
    )


def test_hexbin_column_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(
        HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    )


def test_hexbin_column_plotter_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(
        HexbinColumnPlotter(
            ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
        )
    )


def test_hexbin_column_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(42)


def test_hexbin_column_plotter_equal_nan_true() -> None:
    assert HexbinColumnPlotter(
        ScatterDataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            ),
            x="col1",
            y="col2",
        )
    ).equal(
        HexbinColumnPlotter(
            ScatterDataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                        "col3": [0, 0, 0, 0, 1, 1, 1],
                    }
                ),
                x="col1",
                y="col2",
            )
        ),
        equal_nan=True,
    )


def test_hexbin_column_plotter_equal_nan_false() -> None:
    assert not HexbinColumnPlotter(
        ScatterDataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            ),
            x="col1",
            y="col2",
        )
    ).equal(
        HexbinColumnPlotter(
            ScatterDataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                        "col3": [0, 0, 0, 0, 1, 1, 1],
                    }
                ),
                x="col1",
                y="col2",
            )
        )
    )


def test_hexbin_column_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).plot()
    assert len(figures) == 1
    assert isinstance(figures["hexbin_column"], MatplotlibFigure)


def test_hexbin_column_plotter_plot_empty() -> None:
    figures = HexbinColumnPlotter(
        ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
    ).plot()
    assert len(figures) == 1
    assert figures["hexbin_column"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_hexbin_column_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = HexbinColumnPlotter(ScatterDataFrameState(dataframe, x="col1", y="col2")).plot(
        prefix="prefix_", suffix="_suffix"
    )
    assert len(figures) == 1
    assert isinstance(figures["prefix_hexbin_column_suffix"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(ScatterDataFrameState(dataframe, x="col1", y="col2")),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_1_row() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            ScatterDataFrameState(
                pl.DataFrame({"col1": [0], "col2": [1], "col3": [0]}), x="col1", y="col2"
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_color(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            ScatterDataFrameState(dataframe, x="col1", y="col2", color="col3")
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            ScatterDataFrameState(
                dataframe,
                x="col1",
                y="col2",
                figure_config=MatplotlibFigureConfig(xscale="linear", yscale="linear", init={}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2"))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
