from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.analyzer import ContinuousColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import ContinuousSeriesOutput, Output
from arkas.state import SeriesState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


##############################################
#     Tests for ContinuousColumnAnalyzer     #
##############################################


def test_continuous_column_analyzer_repr() -> None:
    assert repr(ContinuousColumnAnalyzer(column="col1")).startswith("ContinuousColumnAnalyzer(")


def test_continuous_column_analyzer_str() -> None:
    assert str(ContinuousColumnAnalyzer(column="col1")).startswith("ContinuousColumnAnalyzer(")


def test_continuous_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        ContinuousColumnAnalyzer(column="col1")
        .analyze(dataframe)
        .equal(
            ContinuousSeriesOutput(
                SeriesState(
                    pl.Series("col1", [0, 1, 1, 0, 0, 1, 0]),
                )
            )
        )
    )


def test_continuous_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ContinuousColumnAnalyzer(column="col1").analyze(dataframe, lazy=False), Output
    )


def test_continuous_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        ContinuousColumnAnalyzer(column="col1", figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            ContinuousSeriesOutput(
                SeriesState(
                    pl.Series("col1", [0, 1, 1, 0, 0, 1, 0]),
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_continuous_column_analyzer_equal_true() -> None:
    assert ContinuousColumnAnalyzer(column="col1").equal(ContinuousColumnAnalyzer(column="col1"))


def test_continuous_column_analyzer_equal_false_different_column() -> None:
    assert not ContinuousColumnAnalyzer(column="col1").equal(ContinuousColumnAnalyzer(column="col"))


def test_continuous_column_analyzer_equal_false_different_figure_config() -> None:
    assert not ContinuousColumnAnalyzer(
        column="col1", figure_config=MatplotlibFigureConfig(dpi=300)
    ).equal(ContinuousColumnAnalyzer(column="col1", figure_config=MatplotlibFigureConfig()))


def test_continuous_column_analyzer_equal_false_different_type() -> None:
    assert not ContinuousColumnAnalyzer(column="col1").equal(42)


def test_continuous_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        ContinuousColumnAnalyzer(column="col1", figure_config=MatplotlibFigureConfig()).get_args(),
        {
            "column": "col1",
            "figure_config": MatplotlibFigureConfig(),
        },
    )
