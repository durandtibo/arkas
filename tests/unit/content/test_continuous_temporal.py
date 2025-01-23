from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalContinuousColumnContentGenerator
from arkas.content.continuous_temporal import create_template
from arkas.state import TemporalColumnState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 0, 1],
            "col2": [0, 1, 2, 3, 4, 5, 6],
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
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


##############################################################
#     Tests for TemporalContinuousColumnContentGenerator     #
##############################################################


def test_temporal_continuous_column_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    ).startswith("TemporalContinuousColumnContentGenerator(")


def test_temporal_continuous_column_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    ).startswith("TemporalContinuousColumnContentGenerator(")


def test_temporal_continuous_column_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).compute(),
        ContentGenerator,
    )


def test_temporal_continuous_column_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalContinuousColumnContentGenerator(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    )


def test_temporal_continuous_column_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalContinuousColumnContentGenerator(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="col1")
        )
    )


def test_temporal_continuous_column_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalContinuousColumnContentGenerator(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(42)


def test_temporal_continuous_column_content_generator_generate_content(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).generate_content(),
        str,
    )


def test_temporal_continuous_column_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(
                pl.DataFrame({"col2": [], "datetime": []}),
                target_column="col2",
                temporal_column="datetime",
            )
        ).generate_content(),
        str,
    )


def test_temporal_continuous_column_content_generator_generate_body(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).generate_body(),
        str,
    )


def test_temporal_continuous_column_content_generator_generate_body_args(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_temporal_continuous_column_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).generate_toc(),
        str,
    )


def test_temporal_continuous_column_content_generator_generate_toc_args(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        TemporalContinuousColumnContentGenerator(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
