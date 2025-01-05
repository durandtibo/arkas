from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.analyzer import ScatterColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import Output, ScatterColumnOutput
from arkas.state import ScatterDataFrameState


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
#     Tests for ScatterColumnAnalyzer     #
###########################################


def test_scatter_column_analyzer_repr() -> None:
    assert repr(ScatterColumnAnalyzer(x="col1", y="col2")).startswith("ScatterColumnAnalyzer(")


def test_scatter_column_analyzer_str() -> None:
    assert str(ScatterColumnAnalyzer(x="col1", y="col2")).startswith("ScatterColumnAnalyzer(")


def test_scatter_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnAnalyzer(x="col1", y="col2")
        .analyze(dataframe)
        .equal(
            ScatterColumnOutput(
                ScatterDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        }
                    ),
                    x="col1",
                    y="col2",
                )
            )
        )
    )


def test_scatter_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnAnalyzer(x="col1", y="col2").analyze(dataframe, lazy=False), Output
    )


def test_scatter_column_analyzer_analyze_color(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnAnalyzer(x="col1", y="col2", color="col3")
        .analyze(dataframe)
        .equal(
            ScatterColumnOutput(
                ScatterDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                            "col3": [0, 0, 0, 0, 1, 1, 1],
                        }
                    ),
                    x="col1",
                    y="col2",
                    color="col3",
                )
            )
        )
    )


def test_scatter_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnAnalyzer(x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            ScatterColumnOutput(
                ScatterDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [0, 1, 1, 0, 0, 1, 0],
                            "col2": [0, 1, 0, 1, 0, 1, 0],
                        }
                    ),
                    x="col1",
                    y="col2",
                    figure_config=MatplotlibFigureConfig(dpi=50),
                )
            )
        )
    )


def test_scatter_column_analyzer_equal_true() -> None:
    assert ScatterColumnAnalyzer(x="col1", y="col2").equal(
        ScatterColumnAnalyzer(x="col1", y="col2")
    )


def test_scatter_column_analyzer_equal_false_different_x() -> None:
    assert not ScatterColumnAnalyzer(x="col1", y="col2").equal(
        ScatterColumnAnalyzer(x="col", y="col2")
    )


def test_scatter_column_analyzer_equal_false_different_y() -> None:
    assert not ScatterColumnAnalyzer(x="col1", y="col2").equal(
        ScatterColumnAnalyzer(x="col1", y="col")
    )


def test_scatter_column_analyzer_equal_false_different_color() -> None:
    assert not ScatterColumnAnalyzer(x="col1", y="col2", color="col3").equal(
        ScatterColumnAnalyzer(x="col1", y="col2", color="col")
    )


def test_scatter_column_analyzer_equal_false_different_figure_config() -> None:
    assert not ScatterColumnAnalyzer(
        x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=300)
    ).equal(ScatterColumnAnalyzer(x="col1", y="col2", figure_config=MatplotlibFigureConfig()))


def test_scatter_column_analyzer_equal_false_different_type() -> None:
    assert not ScatterColumnAnalyzer(x="col1", y="col2").equal(42)


def test_scatter_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        ScatterColumnAnalyzer(
            x="col1", y="col2", color="col3", figure_config=MatplotlibFigureConfig()
        ).get_args(),
        {
            "x": "col1",
            "y": "col2",
            "color": "col3",
            "figure_config": MatplotlibFigureConfig(),
        },
    )
