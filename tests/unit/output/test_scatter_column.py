from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, ScatterColumnOutput
from arkas.plotter import Plotter
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
#     Tests for ScatterColumnOutput     #
#########################################


def test_scatter_column_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(ScatterColumnOutput(DataFrameState(dataframe))).startswith("ScatterColumnOutput(")


def test_scatter_column_output_str(dataframe: pl.DataFrame) -> None:
    assert str(ScatterColumnOutput(DataFrameState(dataframe))).startswith("ScatterColumnOutput(")


def test_scatter_column_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(ScatterColumnOutput(DataFrameState(dataframe)).compute(), Output)


def test_scatter_column_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert ScatterColumnOutput(DataFrameState(dataframe)).equal(
        ScatterColumnOutput(DataFrameState(dataframe))
    )


def test_scatter_column_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not ScatterColumnOutput(DataFrameState(dataframe)).equal(DataFrameState(pl.DataFrame()))


def test_scatter_column_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ScatterColumnOutput(DataFrameState(dataframe)).equal(42)


def test_scatter_column_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnOutput(DataFrameState(dataframe))
        .get_content_generator()
        .equal(ContentGenerator())
    )


def test_scatter_column_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnOutput(DataFrameState(dataframe)).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_scatter_column_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert ScatterColumnOutput(DataFrameState(dataframe)).get_evaluator().equal(Evaluator())


def test_scatter_column_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnOutput(DataFrameState(dataframe)).get_evaluator(lazy=False).equal(Evaluator())
    )


def test_scatter_column_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    assert ScatterColumnOutput(DataFrameState(dataframe)).get_plotter().equal(Plotter())


def test_scatter_column_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnOutput(DataFrameState(dataframe)).get_plotter(lazy=False), Plotter
    )
