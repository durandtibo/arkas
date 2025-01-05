from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, PlotColumnContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, PlotColumnOutput
from arkas.plotter import PlotColumnPlotter, Plotter
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


######################################
#     Tests for PlotColumnOutput     #
######################################


def test_plot_column_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(PlotColumnOutput(DataFrameState(dataframe))).startswith("PlotColumnOutput(")


def test_plot_column_output_str(dataframe: pl.DataFrame) -> None:
    assert str(PlotColumnOutput(DataFrameState(dataframe))).startswith("PlotColumnOutput(")


def test_plot_column_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotColumnOutput(DataFrameState(dataframe)).compute(), Output)


def test_plot_column_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert PlotColumnOutput(DataFrameState(dataframe)).equal(
        PlotColumnOutput(DataFrameState(dataframe))
    )


def test_plot_column_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not PlotColumnOutput(DataFrameState(dataframe)).equal(DataFrameState(pl.DataFrame()))


def test_plot_column_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not PlotColumnOutput(DataFrameState(dataframe)).equal(42)


def test_plot_column_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnOutput(DataFrameState(dataframe))
        .get_content_generator()
        .equal(PlotColumnContentGenerator(DataFrameState(dataframe)))
    )


def test_plot_column_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        PlotColumnOutput(DataFrameState(dataframe)).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_plot_column_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert PlotColumnOutput(DataFrameState(dataframe)).get_evaluator().equal(Evaluator())


def test_plot_column_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert PlotColumnOutput(DataFrameState(dataframe)).get_evaluator(lazy=False).equal(Evaluator())


def test_plot_column_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        PlotColumnOutput(DataFrameState(dataframe))
        .get_plotter()
        .equal(PlotColumnPlotter(DataFrameState(dataframe)))
    )


def test_plot_column_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotColumnOutput(DataFrameState(dataframe)).get_plotter(lazy=False), Plotter)
