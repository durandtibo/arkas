from __future__ import annotations

import polars as pl
import pytest

from arkas.content import DataFrameSummaryContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import DataFrameSummaryOutput
from arkas.plotter import Plotter


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2, 1, 2.2],
            "int": [1, 1, 0, 1, 1, 1],
            "str": ["A", "B", None, None, "C", "B"],
        },
        schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    )


############################################
#     Tests for DataFrameSummaryOutput     #
############################################


def test_dataframe_summary_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(DataFrameSummaryOutput(dataframe)).startswith("DataFrameSummaryOutput(")


def test_dataframe_summary_output_str(dataframe: pl.DataFrame) -> None:
    assert str(DataFrameSummaryOutput(dataframe)).startswith("DataFrameSummaryOutput(")


def test_dataframe_summary_output_incorrect_top(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'top': -1. The value must be positive"):
        DataFrameSummaryOutput(dataframe, top=-1)


def test_dataframe_summary_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert DataFrameSummaryOutput(dataframe).equal(DataFrameSummaryOutput(dataframe))


def test_dataframe_summary_output_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not DataFrameSummaryOutput(dataframe).equal(
        DataFrameSummaryOutput(
            pl.DataFrame(
                {
                    "float": [1.2, 4.2, None, 2.2, 1, 2.2],
                    "int": [1, 1, 0, 1, 1, 1],
                },
                schema={"float": pl.Float64, "int": pl.Int64},
            )
        )
    )


def test_dataframe_summary_output_equal_false_different_top(dataframe: pl.DataFrame) -> None:
    assert not DataFrameSummaryOutput(dataframe, top=3).equal(DataFrameSummaryOutput(dataframe))


def test_dataframe_summary_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not DataFrameSummaryOutput(dataframe).equal(42)


@pytest.mark.parametrize("top", [1, 2, 3])
def test_dataframe_summary_output_get_content_generator(dataframe: pl.DataFrame, top: int) -> None:
    generator = DataFrameSummaryOutput(dataframe, top=top).get_content_generator()
    assert generator.equal(DataFrameSummaryContentGenerator(dataframe, top=top))


def test_dataframe_summary_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    evaluator = DataFrameSummaryOutput(dataframe).get_evaluator()
    assert evaluator.equal(Evaluator())


def test_dataframe_summary_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    evaluator = DataFrameSummaryOutput(dataframe).get_evaluator(lazy=False)
    assert evaluator.equal(Evaluator())


def test_dataframe_summary_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    plotter = DataFrameSummaryOutput(dataframe).get_plotter()
    assert plotter.equal(Plotter())


def test_dataframe_summary_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    plotter = DataFrameSummaryOutput(dataframe).get_plotter(lazy=False)
    assert plotter.equal(Plotter())