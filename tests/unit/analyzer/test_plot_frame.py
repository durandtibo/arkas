from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import PlotDataFrameAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import Output, PlotDataFrameOutput
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


###########################################
#     Tests for PlotDataFrameAnalyzer     #
###########################################


def test_plot_dataframe_analyzer_repr() -> None:
    assert repr(PlotDataFrameAnalyzer()).startswith("PlotDataFrameAnalyzer(")


def test_plot_dataframe_analyzer_str() -> None:
    assert str(PlotDataFrameAnalyzer()).startswith("PlotDataFrameAnalyzer(")


def test_plot_dataframe_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameAnalyzer()
        .analyze(dataframe)
        .equal(PlotDataFrameOutput(DataFrameState(dataframe)))
    )


def test_plot_dataframe_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotDataFrameAnalyzer().analyze(dataframe, lazy=False), Output)


def test_plot_dataframe_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameAnalyzer(figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            PlotDataFrameOutput(
                DataFrameState(dataframe, figure_config=MatplotlibFigureConfig(dpi=50))
            )
        )
    )


def test_plot_dataframe_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameAnalyzer(columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            PlotDataFrameOutput(
                DataFrameState(
                    pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
                )
            )
        )
    )


def test_plot_dataframe_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameAnalyzer(exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            PlotDataFrameOutput(
                DataFrameState(
                    pl.DataFrame({"col1": [0, 1, 1, 0, 0, 1, 0], "col2": [0, 1, 0, 1, 0, 1, 0]})
                )
            )
        )
    )


def test_plot_dataframe_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotDataFrameAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(PlotDataFrameOutput(DataFrameState(dataframe)))


def test_plot_dataframe_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotDataFrameAnalyzer(columns=["col1", "col2", "col3", "col5"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_plot_dataframe_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = PlotDataFrameAnalyzer(
        columns=["col1", "col2", "col3", "col5"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(PlotDataFrameOutput(DataFrameState(dataframe)))


def test_plot_dataframe_analyzer_equal_true() -> None:
    assert PlotDataFrameAnalyzer().equal(PlotDataFrameAnalyzer())


def test_plot_dataframe_analyzer_equal_false_different_columns() -> None:
    assert not PlotDataFrameAnalyzer().equal(PlotDataFrameAnalyzer(columns=["col1", "col2"]))


def test_plot_dataframe_analyzer_equal_false_different_exclude_columns() -> None:
    assert not PlotDataFrameAnalyzer().equal(PlotDataFrameAnalyzer(exclude_columns=["col2"]))


def test_plot_dataframe_analyzer_equal_false_different_missing_policy() -> None:
    assert not PlotDataFrameAnalyzer().equal(PlotDataFrameAnalyzer(missing_policy="warn"))


def test_plot_dataframe_analyzer_equal_false_different_type() -> None:
    assert not PlotDataFrameAnalyzer().equal(42)


def test_plot_dataframe_analyzer_get_args() -> None:
    assert objects_are_equal(
        PlotDataFrameAnalyzer().get_args(),
        {
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "figure_config": None,
        },
    )
