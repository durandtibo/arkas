from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalNullValueContentGenerator
from arkas.content.temporal_null_value import (
    create_table,
    create_table_row,
    create_template,
)
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


#######################################################
#     Tests for TemporalNullValueContentGenerator     #
#######################################################


def test_temporal_null_value_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        )
    ).startswith("TemporalNullValueContentGenerator(")


def test_temporal_null_value_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        )
    ).startswith("TemporalNullValueContentGenerator(")


def test_temporal_null_value_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).compute(),
        ContentGenerator,
    )


def test_temporal_null_value_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalNullValueContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
    ).equal(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        )
    )


def test_temporal_null_value_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalNullValueContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
    ).equal(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(
                pl.DataFrame(
                    {"col": [], "datetime": []},
                ),
                temporal_column="datetime",
                period="1d",
            )
        )
    )


def test_temporal_null_value_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalNullValueContentGenerator(
        TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
    ).equal(42)


def test_temporal_null_value_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).generate_content(),
        str,
    )


def test_temporal_null_value_content_generator_generate_content_empty() -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(
                pl.DataFrame(
                    {"col": [], "datetime": []},
                ),
                temporal_column="datetime",
                period="1d",
            )
        ).generate_content(),
        str,
    )


def test_temporal_null_value_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).generate_body(),
        str,
    )


def test_temporal_null_value_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_temporal_null_value_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).generate_toc(),
        str,
    )


def test_temporal_null_value_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueContentGenerator(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)


#################################
#    Tests for create_table     #
#################################


def test_create_table(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        create_table(frame=dataframe, temporal_column="datetime", period="1mo"),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(
        create_table(
            frame=pl.DataFrame(
                {"col1": [], "col2": [], "datetime": []},
                schema={
                    "col1": pl.Float64,
                    "col2": pl.Int64,
                    "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
                },
            ),
            temporal_column="datetime",
            period="1mo",
        ),
        str,
    )


#####################################
#    Tests for create_table_row     #
#####################################


def test_create_table_row() -> None:
    assert isinstance(create_table_row(label="meow", num_nulls=5, total=42), str)
