from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import SummaryAnalyzer
from arkas.output import Output, SummaryOutput
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


#####################################
#     Tests for SummaryAnalyzer     #
#####################################


def test_summary_analyzer_analyze_incorrect_top() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'top': -1. The value must be positive"):
        SummaryAnalyzer(top=-1)


def test_summary_analyzer_repr() -> None:
    assert repr(SummaryAnalyzer()).startswith("SummaryAnalyzer(")


def test_summary_analyzer_str() -> None:
    assert str(SummaryAnalyzer()).startswith("SummaryAnalyzer(")


def test_summary_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        SummaryAnalyzer().analyze(dataframe).equal(SummaryOutput(DataFrameState(dataframe, top=5)))
    )


def test_summary_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        SummaryAnalyzer().analyze(dataframe, lazy=False),
        Output,
    )


@pytest.mark.parametrize("top", [0, 1, 2])
def test_summary_analyzer_analyze_top(dataframe: pl.DataFrame, top: int) -> None:
    assert (
        SummaryAnalyzer(top=top)
        .analyze(dataframe)
        .equal(SummaryOutput(DataFrameState(dataframe, top=top)))
    )


def test_summary_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        SummaryAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            SummaryOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        },
                        schema={"col1": pl.Int64, "col2": pl.Int32},
                    ),
                    top=5,
                )
            )
        )
    )


def test_summary_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        SummaryAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            SummaryOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        },
                        schema={"col1": pl.Int64, "col2": pl.Int32},
                    ),
                    top=5,
                )
            )
        )
    )


def test_summary_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = SummaryAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(SummaryOutput(DataFrameState(dataframe, top=5)))


def test_summary_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = SummaryAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_summary_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = SummaryAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(SummaryOutput(DataFrameState(dataframe, top=5)))


def test_summary_analyzer_equal_true() -> None:
    assert SummaryAnalyzer().equal(SummaryAnalyzer())


def test_summary_analyzer_equal_false_different_columns() -> None:
    assert not SummaryAnalyzer().equal(SummaryAnalyzer(columns=["col1", "col2"]))


def test_summary_analyzer_equal_false_different_exclude_columns() -> None:
    assert not SummaryAnalyzer().equal(SummaryAnalyzer(exclude_columns=["col2"]))


def test_summary_analyzer_equal_false_different_missing_policy() -> None:
    assert not SummaryAnalyzer().equal(SummaryAnalyzer(missing_policy="warn"))


def test_summary_analyzer_equal_false_different_top() -> None:
    assert not SummaryAnalyzer().equal(SummaryAnalyzer(top=10))


def test_summary_analyzer_equal_false_different_type() -> None:
    assert not SummaryAnalyzer().equal(42)


def test_summary_analyzer_get_args() -> None:
    assert objects_are_equal(
        SummaryAnalyzer().get_args(),
        {"columns": None, "exclude_columns": (), "missing_policy": "raise", "top": 5},
    )
