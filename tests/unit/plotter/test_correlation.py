from __future__ import annotations

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import CorrelationPlotter, Plotter
from arkas.plotter.correlation import MatplotlibFigureCreator
from arkas.state import TwoColumnDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64},
    )


########################################
#     Tests for CorrelationPlotter     #
########################################


def test_correlation_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        CorrelationPlotter(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationPlotter(")


def test_correlation_plotter_str(dataframe: pl.DataFrame) -> None:
    assert str(
        CorrelationPlotter(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationPlotter(")


def test_correlation_plotter_state(dataframe: pl.DataFrame) -> None:
    assert CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).state.equal(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))


def test_correlation_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationPlotter(
            TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
        ).compute(),
        Plotter,
    )


def test_correlation_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(CorrelationPlotter(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")))


def test_correlation_plotter_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(
        CorrelationPlotter(
            TwoColumnDataFrameState(
                pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
            )
        )
    )


def test_correlation_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(42)


def test_correlation_plotter_equal_nan_true() -> None:
    assert CorrelationPlotter(
        TwoColumnDataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                }
            ),
            column1="col1",
            column2="col2",
        )
    ).equal(
        CorrelationPlotter(
            TwoColumnDataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                    }
                ),
                column1="col1",
                column2="col2",
            )
        ),
        equal_nan=True,
    )


def test_correlation_plotter_equal_nan_false() -> None:
    assert not CorrelationPlotter(
        TwoColumnDataFrameState(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                }
            ),
            column1="col1",
            column2="col2",
        )
    ).equal(
        CorrelationPlotter(
            TwoColumnDataFrameState(
                pl.DataFrame(
                    {
                        "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                        "col2": [0, 1, 0, 1, 0, 1, 0],
                    }
                ),
                column1="col1",
                column2="col2",
            )
        )
    )


def test_correlation_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).plot()
    assert len(figures) == 1
    assert isinstance(figures["correlation"], MatplotlibFigure)


def test_correlation_plotter_plot_empty() -> None:
    figures = CorrelationPlotter(
        TwoColumnDataFrameState(
            pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
        )
    ).plot()
    assert len(figures) == 1
    assert figures["correlation"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_correlation_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = CorrelationPlotter(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_correlation_suffix"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_1_row() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TwoColumnDataFrameState(
                pl.DataFrame({"col1": [0], "col2": [1]}), column1="col1", column2="col2"
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TwoColumnDataFrameState(
                dataframe,
                column1="col1",
                column2="col2",
                figure_config=MatplotlibFigureConfig(xscale="linear", yscale="linear", init={}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(
            TwoColumnDataFrameState(
                pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
            )
        )
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
