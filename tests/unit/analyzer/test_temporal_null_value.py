from __future__ import annotations

import warnings
from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import TemporalNullValueAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import Output, TemporalNullValueOutput
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
#     Tests for TemporalNullValueAnalyzer     #
###############################################


def test_temporal_null_value_analyzer_repr() -> None:
    assert repr(TemporalNullValueAnalyzer(temporal_column="datetime", period="1d")).startswith(
        "TemporalNullValueAnalyzer("
    )


def test_temporal_null_value_analyzer_str() -> None:
    assert str(TemporalNullValueAnalyzer(temporal_column="datetime", period="1d")).startswith(
        "TemporalNullValueAnalyzer("
    )


def test_temporal_null_value_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d")
        .analyze(dataframe)
        .equal(
            TemporalNullValueOutput(
                TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
            )
        )
    )


def test_temporal_null_value_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").analyze(
            dataframe, lazy=False
        ),
        Output,
    )


def test_temporal_null_value_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalNullValueAnalyzer(
            temporal_column="datetime", period="1d", figure_config=MatplotlibFigureConfig(dpi=50)
        )
        .analyze(dataframe)
        .equal(
            TemporalNullValueOutput(
                TemporalDataFrameState(
                    dataframe,
                    temporal_column="datetime",
                    period="1d",
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_temporal_null_value_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d", columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            TemporalNullValueOutput(
                TemporalDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                            "datetime": [
                                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
                            ],
                        }
                    ),
                    temporal_column="datetime",
                    period="1d",
                )
            )
        )
    )


def test_temporal_null_value_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d", exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            TemporalNullValueOutput(
                TemporalDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                            "datetime": [
                                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
                            ],
                        }
                    ),
                    temporal_column="datetime",
                    period="1d",
                )
            )
        )
    )


def test_temporal_null_value_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalNullValueAnalyzer(
        temporal_column="datetime",
        period="1d",
        columns=["col1", "col2", "col3", "col5"],
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(
        TemporalNullValueOutput(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        )
    )


def test_temporal_null_value_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalNullValueAnalyzer(
        temporal_column="datetime", period="1d", columns=["col1", "col2", "col3", "col5"]
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_temporal_null_value_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalNullValueAnalyzer(
        temporal_column="datetime",
        period="1d",
        columns=["col1", "col2", "col3", "col5"],
        missing_policy="warn",
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(
        TemporalNullValueOutput(
            TemporalDataFrameState(dataframe, temporal_column="datetime", period="1d")
        )
    )


def test_temporal_null_value_analyzer_equal_true() -> None:
    assert TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d")
    )


def test_temporal_null_value_analyzer_equal_false_different_temporal_column() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="time", period="1d")
    )


def test_temporal_null_value_analyzer_equal_false_different_period() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1mo")
    )


def test_temporal_null_value_analyzer_equal_false_different_columns() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d", columns=["col1", "col2"])
    )


def test_temporal_null_value_analyzer_equal_false_different_exclude_columns() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d", exclude_columns=["col2"])
    )


def test_temporal_null_value_analyzer_equal_false_different_missing_policy() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d", missing_policy="warn")
    )


def test_temporal_null_value_analyzer_equal_false_different_type() -> None:
    assert not TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").equal(42)


def test_temporal_null_value_analyzer_get_args() -> None:
    assert objects_are_equal(
        TemporalNullValueAnalyzer(temporal_column="datetime", period="1d").get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "figure_config": None,
            "temporal_column": "datetime",
            "period": "1d",
        },
    )
