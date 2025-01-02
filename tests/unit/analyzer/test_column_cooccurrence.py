from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import ColumnCooccurrenceAnalyzer
from arkas.output import ColumnCooccurrenceOutput, Output


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


################################################
#     Tests for ColumnCooccurrenceAnalyzer     #
################################################


def test_column_cooccurrence_analyzer_repr() -> None:
    assert repr(ColumnCooccurrenceAnalyzer()).startswith("ColumnCooccurrenceAnalyzer(")


def test_column_cooccurrence_analyzer_str() -> None:
    assert str(ColumnCooccurrenceAnalyzer()).startswith("ColumnCooccurrenceAnalyzer(")


def test_column_cooccurrence_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCooccurrenceAnalyzer().analyze(dataframe).equal(ColumnCooccurrenceOutput(dataframe))
    )


def test_column_cooccurrence_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrenceAnalyzer().analyze(dataframe, lazy=False), Output)


@pytest.mark.parametrize("ignore_self", [True, False])
def test_column_cooccurrence_analyzer_analyze_ignore_self(
    ignore_self: bool, dataframe: pl.DataFrame
) -> None:
    assert (
        ColumnCooccurrenceAnalyzer(ignore_self=ignore_self)
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                dataframe,
                ignore_self=ignore_self,
            )
        )
    )


def test_column_cooccurrence_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCooccurrenceAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
            )
        )
    )


def test_column_cooccurrence_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCooccurrenceAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
            )
        )
    )


def test_column_cooccurrence_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCooccurrenceAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(ColumnCooccurrenceOutput(dataframe))


def test_column_cooccurrence_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCooccurrenceAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_column_cooccurrence_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCooccurrenceAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(ColumnCooccurrenceOutput(dataframe))


def test_column_cooccurrence_analyzer_equal_true() -> None:
    assert ColumnCooccurrenceAnalyzer().equal(ColumnCooccurrenceAnalyzer())


def test_column_cooccurrence_analyzer_equal_false_different_columns() -> None:
    assert not ColumnCooccurrenceAnalyzer().equal(
        ColumnCooccurrenceAnalyzer(columns=["col1", "col2"])
    )


def test_column_cooccurrence_analyzer_equal_false_different_exclude_columns() -> None:
    assert not ColumnCooccurrenceAnalyzer().equal(
        ColumnCooccurrenceAnalyzer(exclude_columns=["col2"])
    )


def test_column_cooccurrence_analyzer_equal_false_different_missing_policy() -> None:
    assert not ColumnCooccurrenceAnalyzer().equal(ColumnCooccurrenceAnalyzer(missing_policy="warn"))


def test_column_cooccurrence_analyzer_equal_false_different_ignore_self() -> None:
    assert not ColumnCooccurrenceAnalyzer().equal(ColumnCooccurrenceAnalyzer(ignore_self=True))


def test_column_cooccurrence_analyzer_equal_false_different_type() -> None:
    assert not ColumnCooccurrenceAnalyzer().equal(42)


def test_column_cooccurrence_analyzer_get_args() -> None:
    assert objects_are_equal(
        ColumnCooccurrenceAnalyzer().get_args(),
        {"columns": None, "exclude_columns": (), "missing_policy": "raise", "ignore_self": False},
    )
