from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import NullValueAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import NullValueOutput, Output
from arkas.state import NullValueState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, None],
            "col2": [0, 1, None, None, 0, 1, 0],
            "col3": [None, 0, 0, 0, None, 1, None],
        }
    )


#######################################
#     Tests for NullValueAnalyzer     #
#######################################


def test_plot_column_analyzer_repr() -> None:
    assert repr(NullValueAnalyzer()).startswith("NullValueAnalyzer(")


def test_plot_column_analyzer_str() -> None:
    assert str(NullValueAnalyzer()).startswith("NullValueAnalyzer(")


def test_plot_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        NullValueAnalyzer()
        .analyze(dataframe)
        .equal(
            NullValueOutput(
                NullValueState(
                    null_count=np.array([1, 2, 3]),
                    total_count=np.array([7, 7, 7]),
                    columns=["col1", "col2", "col3"],
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(NullValueAnalyzer().analyze(dataframe, lazy=False), Output)


def test_plot_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        NullValueAnalyzer(figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            NullValueOutput(
                NullValueState(
                    null_count=np.array([1, 2, 3]),
                    total_count=np.array([7, 7, 7]),
                    columns=["col1", "col2", "col3"],
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NullValueAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            NullValueOutput(
                NullValueState(
                    null_count=np.array([1, 2]),
                    total_count=np.array([7, 7]),
                    columns=["col1", "col2"],
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        NullValueAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            NullValueOutput(
                NullValueState(
                    null_count=np.array([1, 2]),
                    total_count=np.array([7, 7]),
                    columns=["col1", "col2"],
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NullValueAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_plot_column_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NullValueAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_plot_column_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = NullValueAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_plot_column_analyzer_equal_true() -> None:
    assert NullValueAnalyzer().equal(NullValueAnalyzer())


def test_plot_column_analyzer_equal_false_different_columns() -> None:
    assert not NullValueAnalyzer().equal(NullValueAnalyzer(columns=["col1", "col2"]))


def test_plot_column_analyzer_equal_false_different_exclude_columns() -> None:
    assert not NullValueAnalyzer().equal(NullValueAnalyzer(exclude_columns=["col2"]))


def test_plot_column_analyzer_equal_false_different_missing_policy() -> None:
    assert not NullValueAnalyzer().equal(NullValueAnalyzer(missing_policy="warn"))


def test_plot_column_analyzer_equal_false_different_type() -> None:
    assert not NullValueAnalyzer().equal(42)


def test_plot_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        NullValueAnalyzer().get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "figure_config": None,
        },
    )
