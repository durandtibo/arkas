from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalPlotColumnContentGenerator
from arkas.content.temporal_plot_column import create_template
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


########################################################
#     Tests for TemporalPlotColumnContentGenerator     #
########################################################


def test_plot_column_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    ).startswith("TemporalPlotColumnContentGenerator(")


def test_plot_column_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    ).startswith("TemporalPlotColumnContentGenerator(")


def test_plot_column_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).compute(),
        ContentGenerator,
    )


def test_plot_column_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalPlotColumnContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    )


def test_plot_column_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalPlotColumnContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(
                pl.DataFrame(
                    {"col": [], "datetime": []},
                ),
                temporal_column="datetime",
            )
        )
    )


def test_plot_column_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalPlotColumnContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(42)


def test_plot_column_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_content(),
        str,
    )


def test_plot_column_content_generator_generate_content_empty() -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(
                pl.DataFrame(
                    {"col": [], "datetime": []},
                ),
                temporal_column="datetime",
            )
        ).generate_content(),
        str,
    )


def test_plot_column_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_body(),
        str,
    )


def test_plot_column_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_plot_column_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_toc(),
        str,
    )


def test_plot_column_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
