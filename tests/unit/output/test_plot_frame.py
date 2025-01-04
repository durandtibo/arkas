from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, PlotDataFrameOutput
from arkas.plotter import PlotDataFramePlotter, Plotter
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


#########################################
#     Tests for PlotDataFrameOutput     #
#########################################


def test_plot_dataframe_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(PlotDataFrameOutput(DataFrameState(dataframe))).startswith("PlotDataFrameOutput(")


def test_plot_dataframe_output_str(dataframe: pl.DataFrame) -> None:
    assert str(PlotDataFrameOutput(DataFrameState(dataframe))).startswith("PlotDataFrameOutput(")


def test_plot_dataframe_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(PlotDataFrameOutput(DataFrameState(dataframe)).compute(), Output)


def test_plot_dataframe_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert PlotDataFrameOutput(DataFrameState(dataframe)).equal(
        PlotDataFrameOutput(DataFrameState(dataframe))
    )


def test_plot_dataframe_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not PlotDataFrameOutput(DataFrameState(dataframe)).equal(42)


def test_plot_dataframe_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameOutput(DataFrameState(dataframe))
        .get_content_generator()
        .equal(ContentGenerator())
    )


def test_plot_dataframe_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameOutput(DataFrameState(dataframe))
        .get_content_generator(lazy=False)
        .equal(ContentGenerator())
    )


def test_plot_dataframe_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert PlotDataFrameOutput(DataFrameState(dataframe)).get_evaluator().equal(Evaluator())


def test_plot_dataframe_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameOutput(DataFrameState(dataframe)).get_evaluator(lazy=False).equal(Evaluator())
    )


def test_plot_dataframe_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        PlotDataFrameOutput(DataFrameState(dataframe))
        .get_plotter()
        .equal(PlotDataFramePlotter(DataFrameState(dataframe)))
    )


def test_plot_dataframe_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        PlotDataFrameOutput(DataFrameState(dataframe)).get_plotter(lazy=False), Plotter
    )
