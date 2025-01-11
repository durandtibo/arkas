from __future__ import annotations

import warnings
from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import TemporalPlotColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
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


################################################
#     Tests for TemporalPlotColumnAnalyzer     #
################################################


def test_temporal_plot_column_analyzer_repr() -> None:
    assert repr(TemporalPlotColumnAnalyzer(temporal_column="datetime")).startswith(
        "TemporalPlotColumnAnalyzer("
    )


def test_temporal_plot_column_analyzer_str() -> None:
    assert str(TemporalPlotColumnAnalyzer(temporal_column="datetime")).startswith(
        "TemporalPlotColumnAnalyzer("
    )


def test_temporal_plot_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnAnalyzer(temporal_column="datetime")
        .analyze(dataframe)
        .equal(
            TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
        )
    )


def test_temporal_plot_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        TemporalPlotColumnAnalyzer(temporal_column="datetime").analyze(dataframe, lazy=False),
        Output,
    )


def test_temporal_plot_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnAnalyzer(
            temporal_column="datetime", figure_config=MatplotlibFigureConfig(dpi=50)
        )
        .analyze(dataframe)
        .equal(
            TemporalPlotColumnOutput(
                TemporalDataFrameState(
                    dataframe,
                    temporal_column="datetime",
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_temporal_plot_column_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnAnalyzer(temporal_column="datetime", columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            TemporalPlotColumnOutput(
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
                )
            )
        )
    )


def test_temporal_plot_column_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalPlotColumnAnalyzer(temporal_column="datetime", exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            TemporalPlotColumnOutput(
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
                )
            )
        )
    )


def test_temporal_plot_column_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalPlotColumnAnalyzer(
        temporal_column="datetime",
        columns=["col1", "col2", "col3", "col5"],
        missing_policy="ignore",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    )


def test_temporal_plot_column_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalPlotColumnAnalyzer(
        temporal_column="datetime", columns=["col1", "col2", "col3", "col5"]
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_temporal_plot_column_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = TemporalPlotColumnAnalyzer(
        temporal_column="datetime", columns=["col1", "col2", "col3", "col5"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(
        TemporalPlotColumnOutput(TemporalDataFrameState(dataframe, temporal_column="datetime"))
    )


def test_temporal_plot_column_analyzer_equal_true() -> None:
    assert TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime")
    )


def test_temporal_plot_column_analyzer_equal_false_different_temporal_column() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="time")
    )


def test_temporal_plot_column_analyzer_equal_false_different_period() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime", period="1mo")
    )


def test_temporal_plot_column_analyzer_equal_false_different_columns() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime", columns=["col1", "col2"])
    )


def test_temporal_plot_column_analyzer_equal_false_different_exclude_columns() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime", exclude_columns=["col2"])
    )


def test_temporal_plot_column_analyzer_equal_false_different_missing_policy() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime", missing_policy="warn")
    )


def test_temporal_plot_column_analyzer_equal_false_different_type() -> None:
    assert not TemporalPlotColumnAnalyzer(temporal_column="datetime").equal(42)


def test_temporal_plot_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        TemporalPlotColumnAnalyzer(temporal_column="datetime").get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "figure_config": None,
            "temporal_column": "datetime",
            "period": None,
        },
    )
