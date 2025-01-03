from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ColumnCooccurrenceContentGenerator, ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.figure import MatplotlibFigureConfig
from arkas.output import ColumnCooccurrenceOutput, Output
from arkas.plotter import Plotter


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


############################################
#     Tests for ColumnCooccurrenceOutput     #
############################################


def test_column_cooccurrence_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(ColumnCooccurrenceOutput(dataframe)).startswith("ColumnCooccurrenceOutput(")


def test_column_cooccurrence_output_str(dataframe: pl.DataFrame) -> None:
    assert str(ColumnCooccurrenceOutput(dataframe)).startswith("ColumnCooccurrenceOutput(")


def test_balanced_accuracy_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCooccurrenceOutput(dataframe).compute(),
        Output,
    )


def test_column_cooccurrence_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceOutput(dataframe).equal(ColumnCooccurrenceOutput(dataframe))


def test_column_cooccurrence_output_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not ColumnCooccurrenceOutput(dataframe).equal(
        ColumnCooccurrenceOutput(
            pl.DataFrame(
                {
                    "float": [1.2, 4.2, None, 2.2, 1, 2.2],
                    "int": [1, 1, 0, 1, 1, 1],
                },
                schema={"float": pl.Float64, "int": pl.Int64},
            )
        )
    )


def test_column_cooccurrence_output_equal_false_different_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceOutput(dataframe, ignore_self=True).equal(
        ColumnCooccurrenceOutput(dataframe)
    )


def test_column_cooccurrence_output_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceOutput(dataframe).equal(
        ColumnCooccurrenceOutput(dataframe, figure_config=MatplotlibFigureConfig(dpi=50))
    )


def test_column_cooccurrence_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ColumnCooccurrenceOutput(dataframe).equal(42)


@pytest.mark.parametrize("ignore_self", [True, False])
def test_column_cooccurrence_output_get_content_generator_lazy_true(
    dataframe: pl.DataFrame, ignore_self: bool
) -> None:
    assert (
        ColumnCooccurrenceOutput(dataframe, ignore_self=ignore_self)
        .get_content_generator()
        .equal(ColumnCooccurrenceContentGenerator(dataframe, ignore_self=ignore_self))
    )


@pytest.mark.parametrize("ignore_self", [True, False])
def test_column_cooccurrence_output_get_content_generator_lazy_false(
    dataframe: pl.DataFrame, ignore_self: bool
) -> None:
    assert isinstance(
        ColumnCooccurrenceOutput(dataframe, ignore_self=ignore_self).get_content_generator(
            lazy=False
        ),
        ContentGenerator,
    )


def test_column_cooccurrence_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceOutput(dataframe).get_evaluator().equal(Evaluator())


def test_column_cooccurrence_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceOutput(dataframe).get_evaluator(lazy=False).equal(Evaluator())


def test_column_cooccurrence_output_get_plotter_lazy_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceOutput(dataframe).get_plotter().equal(Plotter())


def test_column_cooccurrence_output_get_plotter_lazy_false(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceOutput(dataframe).get_plotter(lazy=False).equal(Plotter())
