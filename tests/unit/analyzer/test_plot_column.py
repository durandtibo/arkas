from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import PlotColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import Output, PlotColumnOutput
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


########################################
#     Tests for PlotColumnAnalyzer     #
########################################


def test_plot_column_analyzer_repr() -> None:
    assert repr(PlotColumnAnalyzer()).startswith("PlotColumnAnalyzer(")


def test_plot_column_analyzer_str() -> None:
    assert str(PlotColumnAnalyzer()).startswith("PlotColumnAnalyzer(")


def test_plot_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnAnalyzer().analyze(dataframe).equal(PlotColumnOutput(DataFrameState(dataframe)))
    )


def test_plot_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotColumnAnalyzer().analyze(dataframe, lazy=False), Output)


def test_plot_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnAnalyzer(figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            PlotColumnOutput(
                DataFrameState(dataframe, figure_config=MatplotlibFigureConfig(dpi=50))
            )
        )
    )


def test_plot_column_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            PlotColumnOutput(
                DataFrameState(
                    pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            PlotColumnOutput(
                DataFrameState(
                    pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
                )
            )
        )
    )


def test_plot_column_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotColumnAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(PlotColumnOutput(DataFrameState(dataframe)))


def test_plot_column_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotColumnAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_plot_column_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotColumnAnalyzer(columns=["col1", "col2", "col3", "col5"], missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(PlotColumnOutput(DataFrameState(dataframe)))


def test_plot_column_analyzer_equal_true() -> None:
    assert PlotColumnAnalyzer().equal(PlotColumnAnalyzer())


def test_plot_column_analyzer_equal_false_different_columns() -> None:
    assert not PlotColumnAnalyzer().equal(PlotColumnAnalyzer(columns=["col1", "col2"]))


def test_plot_column_analyzer_equal_false_different_exclude_columns() -> None:
    assert not PlotColumnAnalyzer().equal(PlotColumnAnalyzer(exclude_columns=["col2"]))


def test_plot_column_analyzer_equal_false_different_missing_policy() -> None:
    assert not PlotColumnAnalyzer().equal(PlotColumnAnalyzer(missing_policy="warn"))


def test_plot_column_analyzer_equal_false_different_type() -> None:
    assert not PlotColumnAnalyzer().equal(42)


def test_plot_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        PlotColumnAnalyzer().get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "figure_config": None,
        },
    )
