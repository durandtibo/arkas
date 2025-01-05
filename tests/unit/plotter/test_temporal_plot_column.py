from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import Plotter, TemporalPlotColumnPlotter
from arkas.plotter.temporal_plot_column import MatplotlibFigureCreator
from arkas.state import TemporalDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
            "datetime": [
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Int64,
            "col3": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


###############################################
#     Tests for TemporalPlotColumnPlotter     #
###############################################


def test_temporal_plot_column_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalPlotColumnPlotter(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    ).startswith("TemporalPlotColumnPlotter(")


def test_temporal_plot_column_plotter_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalPlotColumnPlotter(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    ).startswith("TemporalPlotColumnPlotter(")


def test_temporal_plot_column_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnPlotter(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).compute(),
        Plotter,
    )


def test_temporal_plot_column_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalPlotColumnPlotter(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalPlotColumnPlotter(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    )


def test_temporal_plot_column_plotter_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not TemporalPlotColumnPlotter(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalPlotColumnPlotter(
            TemporalDataFrameState(pl.DataFrame({"datetime": []}), temporal_column="datetime")
        )
    )


def test_temporal_plot_column_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TemporalPlotColumnPlotter(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(42)


def test_temporal_plot_column_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = TemporalPlotColumnPlotter(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).plot()
    assert len(figures) == 1
    assert isinstance(figures["temporal_plot_column"], MatplotlibFigure)


def test_temporal_plot_column_plotter_plot_empty() -> None:
    figures = TemporalPlotColumnPlotter(
        TemporalDataFrameState(pl.DataFrame({"datetime": []}), temporal_column="datetime")
    ).plot()
    assert len(figures) == 1
    assert figures["temporal_plot_column"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_temporal_plot_column_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = TemporalPlotColumnPlotter(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_temporal_plot_column_suffix"], MatplotlibFigure)


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
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_with_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TemporalDataFrameState(
                dataframe,
                temporal_column="datetime",
                figure_config=MatplotlibFigureConfig(yscale="symlog", init={"figsize": (3, 3)}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(TemporalDataFrameState(pl.DataFrame({"datetime": []}), temporal_column="datetime"))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
