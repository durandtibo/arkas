from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalPlotColumnContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, TemporalPlotColumnOutput
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


##############################################
#     Tests for TemporalPlotColumnOutput     #
##############################################


def test_temporal_plot_column_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    ).startswith("TemporalPlotColumnOutput(")


def test_temporal_plot_column_output_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    ).startswith("TemporalPlotColumnOutput(")


def test_temporal_plot_column_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnOutput(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).compute(),
        Output,
    )


def test_temporal_plot_column_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalPlotColumnOutput(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime")))


def test_temporal_plot_column_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not TemporalPlotColumnOutput(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="col1")))


def test_temporal_plot_column_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TemporalPlotColumnOutput(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(42)


def test_temporal_plot_column_output_get_content_generator_lazy_true(
    dataframe: pl.DataFrame,
) -> None:
    assert (
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
        .get_content_generator()
        .equal(
            TemporalPlotColumnContentGenerator(
                TemporalDataFrameState(dataframe, temporal_column="datetime")
            )
        )
    )


def test_temporal_plot_column_output_get_content_generator_lazy_false(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalPlotColumnOutput(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_temporal_plot_column_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
        .get_evaluator()
        .equal(Evaluator())
    )


def test_temporal_plot_column_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
        .get_evaluator(lazy=False)
        .equal(Evaluator())
    )
