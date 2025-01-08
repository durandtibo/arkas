from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import NumericSummaryAnalyzer
from arkas.output import NumericSummaryOutput, Output
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    )


############################################
#     Tests for NumericSummaryAnalyzer     #
############################################


def test_numeric_summary_analyzer_repr() -> None:
    assert repr(NumericSummaryAnalyzer()).startswith("NumericSummaryAnalyzer(")


def test_numeric_summary_analyzer_str() -> None:
    assert str(NumericSummaryAnalyzer()).startswith("NumericSummaryAnalyzer(")


def test_numeric_summary_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer()
        .analyze(dataframe)
        .equal(NumericSummaryOutput(DataFrameState(dataframe)))
    )


def test_numeric_summary_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(NumericSummaryAnalyzer().analyze(dataframe, lazy=False), Output)


def test_numeric_summary_analyzer_analyze_ignore_non_numeric_columns(
    dataframe: pl.DataFrame,
) -> None:
    assert (
        NumericSummaryAnalyzer()
        .analyze(dataframe.with_columns(pl.lit("abc").alias("col4")))
        .equal(NumericSummaryOutput(DataFrameState(dataframe)))
    )


def test_numeric_summary_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            NumericSummaryOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        },
                        schema={"col1": pl.Int64, "col2": pl.Int32},
                    )
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            NumericSummaryOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        },
                        schema={"col1": pl.Int64, "col2": pl.Int32},
                    )
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NumericSummaryAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(NumericSummaryOutput(DataFrameState(dataframe)))


def test_numeric_summary_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NumericSummaryAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_numeric_summary_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NumericSummaryAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(NumericSummaryOutput(DataFrameState(dataframe)))


def test_numeric_summary_analyzer_equal_true() -> None:
    assert NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer())


def test_numeric_summary_analyzer_equal_false_different_columns() -> None:
    assert not NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer(columns=["col1", "col2"]))


def test_numeric_summary_analyzer_equal_false_different_exclude_columns() -> None:
    assert not NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer(exclude_columns=["col2"]))


def test_numeric_summary_analyzer_equal_false_different_missing_policy() -> None:
    assert not NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer(missing_policy="warn"))


def test_numeric_summary_analyzer_equal_false_different_type() -> None:
    assert not NumericSummaryAnalyzer().equal(42)


def test_numeric_summary_analyzer_get_args() -> None:
    assert objects_are_equal(
        NumericSummaryAnalyzer().get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
        },
    )
