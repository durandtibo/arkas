from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import NumericSummaryAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import ColumnCooccurrenceOutput, Output
from arkas.state import ColumnCooccurrenceState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
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
        .equal(
            ColumnCooccurrenceOutput(
                ColumnCooccurrenceState(
                    matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                    columns=["col1", "col2", "col3"],
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(NumericSummaryAnalyzer().analyze(dataframe, lazy=False), Output)


def test_numeric_summary_analyzer_analyze_ignore_self_true(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(ignore_self=True)
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                ColumnCooccurrenceState(
                    matrix=np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
                    columns=["col1", "col2", "col3"],
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                ColumnCooccurrenceState(
                    matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                    columns=["col1", "col2", "col3"],
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                ColumnCooccurrenceState(
                    matrix=np.array([[3, 2], [2, 3]], dtype=int), columns=["col1", "col2"]
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            ColumnCooccurrenceOutput(
                ColumnCooccurrenceState(
                    matrix=np.array([[3, 2], [2, 3]], dtype=int), columns=["col1", "col2"]
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
    assert out.equal(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    )


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
    assert out.equal(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_numeric_summary_analyzer_equal_true() -> None:
    assert NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer())


def test_numeric_summary_analyzer_equal_false_different_columns() -> None:
    assert not NumericSummaryAnalyzer().equal(
        NumericSummaryAnalyzer(columns=["col1", "col2"])
    )


def test_numeric_summary_analyzer_equal_false_different_exclude_columns() -> None:
    assert not NumericSummaryAnalyzer().equal(
        NumericSummaryAnalyzer(exclude_columns=["col2"])
    )


def test_numeric_summary_analyzer_equal_false_different_missing_policy() -> None:
    assert not NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer(missing_policy="warn"))


def test_numeric_summary_analyzer_equal_false_different_ignore_self() -> None:
    assert not NumericSummaryAnalyzer().equal(NumericSummaryAnalyzer(ignore_self=True))


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
