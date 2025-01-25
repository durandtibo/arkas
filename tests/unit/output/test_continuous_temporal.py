from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from arkas.content import ContentGenerator, TemporalContinuousColumnContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, TemporalContinuousColumnOutput
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


####################################################
#     Tests for TemporalContinuousColumnOutput     #
####################################################


def test_continuous_series_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    ).startswith("TemporalContinuousColumnOutput(")


def test_continuous_series_output_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    ).startswith("TemporalContinuousColumnOutput(")


def test_continuous_series_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).compute(),
        Output,
    )


def test_continuous_series_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalContinuousColumnOutput(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
    )


def test_continuous_series_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not TemporalContinuousColumnOutput(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="col1")
        )
    )


def test_continuous_series_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TemporalContinuousColumnOutput(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).equal(42)


def test_continuous_series_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
        .get_content_generator()
        .equal(
            TemporalContinuousColumnContentGenerator(
                TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
            )
        )
    )


def test_continuous_series_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_continuous_series_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
        .get_evaluator()
        .equal(Evaluator())
    )


def test_continuous_series_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnOutput(
            TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
        )
        .get_evaluator(lazy=False)
        .equal(Evaluator())
    )
