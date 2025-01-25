from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.analyzer import TemporalContinuousColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
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


######################################################
#     Tests for TemporalContinuousColumnAnalyzer     #
######################################################


def test_temporal_continuous_column_analyzer_repr() -> None:
    assert repr(
        TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime")
    ).startswith("TemporalContinuousColumnAnalyzer(")


def test_temporal_continuous_column_analyzer_str() -> None:
    assert str(
        TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime")
    ).startswith("TemporalContinuousColumnAnalyzer(")


def test_temporal_continuous_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime")
        .analyze(dataframe)
        .equal(
            TemporalContinuousColumnOutput(
                TemporalColumnState(
                    dataframe,
                    target_column="col2",
                    temporal_column="datetime",
                    period=None,
                    nan_policy="propagate",
                    figure_config=None,
                )
            )
        )
    )


def test_temporal_continuous_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime").analyze(
            dataframe, lazy=False
        ),
        Output,
    )


def test_temporal_continuous_column_analyzer_analyze_period(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnAnalyzer(
            target_column="col2",
            temporal_column="datetime",
            period="1mo",
        )
        .analyze(dataframe)
        .equal(
            TemporalContinuousColumnOutput(
                TemporalColumnState(
                    dataframe,
                    target_column="col2",
                    temporal_column="datetime",
                    period="1mo",
                    nan_policy="propagate",
                    figure_config=MatplotlibFigureConfig(),
                )
            )
        )
    )


def test_temporal_continuous_column_analyzer_analyze_nan_policy(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnAnalyzer(
            target_column="col2",
            temporal_column="datetime",
            nan_policy="raise",
        )
        .analyze(dataframe)
        .equal(
            TemporalContinuousColumnOutput(
                TemporalColumnState(
                    dataframe,
                    target_column="col2",
                    temporal_column="datetime",
                    period=None,
                    nan_policy="raise",
                    figure_config=MatplotlibFigureConfig(),
                )
            )
        )
    )


def test_temporal_continuous_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalContinuousColumnAnalyzer(
            target_column="col2",
            temporal_column="datetime",
            figure_config=MatplotlibFigureConfig(dpi=50),
        )
        .analyze(dataframe)
        .equal(
            TemporalContinuousColumnOutput(
                TemporalColumnState(
                    dataframe,
                    target_column="col2",
                    temporal_column="datetime",
                    period=None,
                    nan_policy="propagate",
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_temporal_continuous_column_analyzer_equal_true() -> None:
    assert TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime").equal(
        TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="datetime")
    )


def test_temporal_continuous_column_analyzer_equal_false_different_target_column() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2", temporal_column="datetime"
    ).equal(TemporalContinuousColumnAnalyzer(target_column="col1", temporal_column="datetime"))


def test_temporal_continuous_column_analyzer_equal_false_different_temporal_column() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2", temporal_column="datetime"
    ).equal(TemporalContinuousColumnAnalyzer(target_column="col2", temporal_column="time"))


def test_temporal_continuous_column_analyzer_equal_false_different_period() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalContinuousColumnAnalyzer(
            target_column="col2", temporal_column="datetime", period="1mo"
        )
    )


def test_temporal_continuous_column_analyzer_equal_false_different_nan_policy() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalContinuousColumnAnalyzer(
            target_column="col2", temporal_column="datetime", nan_policy="raise"
        )
    )


def test_temporal_continuous_column_analyzer_equal_false_different_figure_config() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2",
        temporal_column="datetime",
        figure_config=MatplotlibFigureConfig(dpi=300),
    ).equal(
        TemporalContinuousColumnAnalyzer(
            target_column="col2", temporal_column="datetime", figure_config=MatplotlibFigureConfig()
        )
    )


def test_temporal_continuous_column_analyzer_equal_false_different_type() -> None:
    assert not TemporalContinuousColumnAnalyzer(
        target_column="col2", temporal_column="datetime"
    ).equal(42)


def test_temporal_continuous_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        TemporalContinuousColumnAnalyzer(
            target_column="col2", temporal_column="datetime", figure_config=MatplotlibFigureConfig()
        ).get_args(),
        {
            "target_column": "col2",
            "temporal_column": "datetime",
            "period": None,
            "nan_policy": "propagate",
            "figure_config": MatplotlibFigureConfig(),
        },
    )
