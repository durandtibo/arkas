from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, ScatterColumnContentGenerator
from arkas.evaluator2 import Evaluator
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


#########################################
#     Tests for ScatterColumnOutput     #
#########################################


def test_scatter_column_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("ScatterColumnOutput(")


def test_scatter_column_output_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("ScatterColumnOutput(")


def test_scatter_column_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2")).compute(), Output
    )


def test_scatter_column_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    )


def test_scatter_column_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(
        ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
    )


def test_scatter_column_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2")).equal(42)


def test_scatter_column_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
        .get_content_generator()
        .equal(ScatterColumnContentGenerator(ScatterDataFrameState(dataframe, x="col1", y="col2")))
    )


def test_scatter_column_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnOutput(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_scatter_column_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
        .get_evaluator()
        .equal(Evaluator())
    )


def test_scatter_column_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterColumnOutput(ScatterDataFrameState(dataframe, x="col1", y="col2"))
        .get_evaluator(lazy=False)
        .equal(Evaluator())
    )
