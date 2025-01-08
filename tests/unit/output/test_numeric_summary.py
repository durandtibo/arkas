from __future__ import annotations

import polars as pl
import pytest

from arkas.content import (
    ContentGenerator,
    NumericSummaryContentGenerator,
)
from arkas.evaluator2 import Evaluator
from arkas.output import NumericSummaryOutput, Output
from arkas.plotter import Plotter
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    )


##########################################
#     Tests for NumericSummaryOutput     #
##########################################


def test_numeric_summary_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(NumericSummaryOutput(DataFrameState(dataframe))).startswith("NumericSummaryOutput(")


def test_numeric_summary_output_str(dataframe: pl.DataFrame) -> None:
    assert str(NumericSummaryOutput(DataFrameState(dataframe))).startswith("NumericSummaryOutput(")


def test_numeric_summary_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(NumericSummaryOutput(DataFrameState(dataframe)).compute(), Output)


def test_numeric_summary_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert NumericSummaryOutput(DataFrameState(dataframe)).equal(
        NumericSummaryOutput(DataFrameState(dataframe))
    )


def test_numeric_summary_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not NumericSummaryOutput(DataFrameState(dataframe)).equal(DataFrameState(pl.DataFrame()))


def test_numeric_summary_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not NumericSummaryOutput(DataFrameState(dataframe)).equal(42)


def test_numeric_summary_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryOutput(DataFrameState(dataframe))
        .get_content_generator()
        .equal(NumericSummaryContentGenerator(DataFrameState(dataframe)))
    )


def test_numeric_summary_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryOutput(DataFrameState(dataframe)).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_numeric_summary_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert NumericSummaryOutput(DataFrameState(dataframe)).get_evaluator().equal(Evaluator())


def test_numeric_summary_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryOutput(DataFrameState(dataframe)).get_evaluator(lazy=False).equal(Evaluator())
    )


def test_numeric_summary_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    assert NumericSummaryOutput(DataFrameState(dataframe)).get_plotter().equal(Plotter())


def test_numeric_summary_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryOutput(DataFrameState(dataframe)).get_plotter(lazy=False), Plotter
    )
