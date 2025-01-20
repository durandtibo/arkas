from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalContinuousSeriesContentGenerator
from arkas.content.continuous_temporal import create_template
from arkas.state import TemporalDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 2, 3, 4, 5, 6],
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
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


##############################################################
#     Tests for TemporalContinuousSeriesContentGenerator     #
##############################################################


def test_temporal_continuous_series_content_generator_incorrect_state(
    dataframe: pl.DataFrame,
) -> None:
    with pytest.raises(
        ValueError, match="The DataFrame must have 2 columns but received a DataFrame of shape"
    ):
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(
                dataframe.with_columns(pl.lit(1).alias("col2")), temporal_column="datetime"
            )
        )


def test_temporal_continuous_series_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    ).startswith("TemporalContinuousSeriesContentGenerator(")


def test_temporal_continuous_series_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    ).startswith("TemporalContinuousSeriesContentGenerator(")


def test_temporal_continuous_series_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).compute(),
        ContentGenerator,
    )


def test_temporal_continuous_series_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalContinuousSeriesContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        )
    )


def test_temporal_continuous_series_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalContinuousSeriesContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="col1")
        )
    )


def test_temporal_continuous_series_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalContinuousSeriesContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime")
    ).equal(42)


def test_temporal_continuous_series_content_generator_generate_content(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_content(),
        str,
    )


def test_temporal_continuous_series_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(
                pl.DataFrame({"col": [], "datetime": []}), temporal_column="datetime"
            )
        ).generate_content(),
        str,
    )


def test_temporal_continuous_series_content_generator_generate_body(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_body(),
        str,
    )


def test_temporal_continuous_series_content_generator_generate_body_args(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_temporal_continuous_series_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_toc(),
        str,
    )


def test_temporal_continuous_series_content_generator_generate_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousSeriesContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
