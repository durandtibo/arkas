from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import Plotter, TemporalPlotColumnPlotter
from arkas.plotter.temporal_plot_column import MatplotlibFigureCreator, prepare_data
from arkas.state import TemporalDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [2, 3, 4, 5, 6, 7, 1],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
            "datetime": [
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
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


def test_matplotlib_figure_creator_create_1_row() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TemporalDataFrameState(
                pl.DataFrame(
                    {
                        "col1": [2],
                        "col2": [0],
                        "col3": [0],
                        "datetime": [datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)],
                    },
                    schema={
                        "col1": pl.Int64,
                        "col2": pl.Int64,
                        "col3": pl.Int64,
                        "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                    },
                ),
                temporal_column="datetime",
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            TemporalDataFrameState(
                dataframe,
                temporal_column="datetime",
                figure_config=MatplotlibFigureConfig(yscale="symlog", init={}),
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


##################################
#     Tests for prepare_data     #
##################################


@pytest.mark.filterwarnings(
    "ignore:no explicit representation of timezones available for np.datetime64"
)
def test_prepare_data_no_period(dataframe: pl.DataFrame) -> None:
    data, time = prepare_data(dataframe, temporal_column="datetime", period=None)
    assert objects_are_equal(
        data,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5, 6, 7],
                "col2": [0, 0, 1, 0, 1, 0, 1],
                "col3": [1, 0, 0, 0, 0, 1, 1],
            },
            schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
        ),
    )
    assert objects_are_equal(
        time,
        np.array(
            [
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
            ],
            dtype="datetime64[us]",
        ),
    )


@pytest.mark.filterwarnings(
    "ignore:no explicit representation of timezones available for np.datetime64"
)
def test_prepare_data_period_1d(dataframe: pl.DataFrame) -> None:
    data, time = prepare_data(dataframe, temporal_column="datetime", period="1d")
    assert objects_are_equal(
        data,
        pl.DataFrame(
            {
                "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "col2": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "col3": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            },
            schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
        ),
    )
    assert objects_are_equal(
        time,
        np.array(
            [
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
            ],
            dtype="datetime64[us]",
        ),
    )


@pytest.mark.filterwarnings(
    "ignore:no explicit representation of timezones available for np.datetime64"
)
def test_prepare_data_period_2d(dataframe: pl.DataFrame) -> None:
    data, time = prepare_data(dataframe, temporal_column="datetime", period="2d")
    assert objects_are_equal(
        data,
        pl.DataFrame(
            {
                "col1": [1.5, 3.5, 5.5, 7.0],
                "col2": [0.0, 0.5, 0.5, 1.0],
                "col3": [0.5, 0.0, 0.5, 1.0],
            },
            schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
        ),
    )
    assert objects_are_equal(
        time,
        np.array(
            [
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
            ],
            dtype="datetime64[us]",
        ),
    )
