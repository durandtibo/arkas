from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.analyzer import HexbinColumnAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import HexbinColumnOutput, Output
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


##########################################
#     Tests for HexbinColumnAnalyzer     #
##########################################


def test_hexbin_column_analyzer_repr() -> None:
    assert repr(HexbinColumnAnalyzer(x="col1", y="col2")).startswith("HexbinColumnAnalyzer(")


def test_hexbin_column_analyzer_str() -> None:
    assert str(HexbinColumnAnalyzer(x="col1", y="col2")).startswith("HexbinColumnAnalyzer(")


def test_hexbin_column_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        HexbinColumnAnalyzer(x="col1", y="col2")
        .analyze(dataframe)
        .equal(
            HexbinColumnOutput(
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


def test_hexbin_column_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        HexbinColumnAnalyzer(x="col1", y="col2").analyze(dataframe, lazy=False), Output
    )


def test_hexbin_column_analyzer_analyze_color(dataframe: pl.DataFrame) -> None:
    assert (
        HexbinColumnAnalyzer(x="col1", y="col2", color="col3")
        .analyze(dataframe)
        .equal(
            HexbinColumnOutput(
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


def test_hexbin_column_analyzer_analyze_figure_config(dataframe: pl.DataFrame) -> None:
    assert (
        HexbinColumnAnalyzer(x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=50))
        .analyze(dataframe)
        .equal(
            HexbinColumnOutput(
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


def test_hexbin_column_analyzer_equal_true() -> None:
    assert HexbinColumnAnalyzer(x="col1", y="col2").equal(HexbinColumnAnalyzer(x="col1", y="col2"))


def test_hexbin_column_analyzer_equal_false_different_x() -> None:
    assert not HexbinColumnAnalyzer(x="col1", y="col2").equal(
        HexbinColumnAnalyzer(x="col", y="col2")
    )


def test_hexbin_column_analyzer_equal_false_different_y() -> None:
    assert not HexbinColumnAnalyzer(x="col1", y="col2").equal(
        HexbinColumnAnalyzer(x="col1", y="col")
    )


def test_hexbin_column_analyzer_equal_false_different_color() -> None:
    assert not HexbinColumnAnalyzer(x="col1", y="col2", color="col3").equal(
        HexbinColumnAnalyzer(x="col1", y="col2", color="col")
    )


def test_hexbin_column_analyzer_equal_false_different_figure_config() -> None:
    assert not HexbinColumnAnalyzer(
        x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=300)
    ).equal(HexbinColumnAnalyzer(x="col1", y="col2", figure_config=MatplotlibFigureConfig()))


def test_hexbin_column_analyzer_equal_false_different_type() -> None:
    assert not HexbinColumnAnalyzer(x="col1", y="col2").equal(42)


def test_hexbin_column_analyzer_get_args() -> None:
    assert objects_are_equal(
        HexbinColumnAnalyzer(
            x="col1", y="col2", color="col3", figure_config=MatplotlibFigureConfig()
        ).get_args(),
        {
            "x": "col1",
            "y": "col2",
            "color": "col3",
            "figure_config": MatplotlibFigureConfig(),
        },
    )
