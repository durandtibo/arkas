from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, SummaryContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output, SummaryOutput


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


###################################
#     Tests for SummaryOutput     #
###################################


def test_summary_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(SummaryOutput(dataframe)).startswith("SummaryOutput(")


def test_summary_output_str(dataframe: pl.DataFrame) -> None:
    assert str(SummaryOutput(dataframe)).startswith("SummaryOutput(")


def test_summary_output_incorrect_top(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'top': -1. The value must be positive"):
        SummaryOutput(dataframe, top=-1)


def test_summary_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        SummaryOutput(dataframe).compute(),
        Output,
    )


def test_summary_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert SummaryOutput(dataframe).equal(SummaryOutput(dataframe))


def test_summary_output_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not SummaryOutput(dataframe).equal(
        SummaryOutput(
            pl.DataFrame(
                {
                    "float": [1.2, 4.2, None, 2.2, 1, 2.2],
                    "int": [1, 1, 0, 1, 1, 1],
                },
                schema={"float": pl.Float64, "int": pl.Int64},
            )
        )
    )


def test_summary_output_equal_false_different_top(dataframe: pl.DataFrame) -> None:
    assert not SummaryOutput(dataframe, top=3).equal(SummaryOutput(dataframe))


def test_summary_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not SummaryOutput(dataframe).equal(42)


@pytest.mark.parametrize("top", [1, 2, 3])
def test_summary_output_get_content_generator_lazy_true(dataframe: pl.DataFrame, top: int) -> None:
    assert (
        SummaryOutput(dataframe, top=top)
        .get_content_generator()
        .equal(SummaryContentGenerator(dataframe, top=top))
    )


@pytest.mark.parametrize("top", [1, 2, 3])
def test_summary_output_get_content_generator_lazy_false(dataframe: pl.DataFrame, top: int) -> None:
    assert isinstance(
        SummaryOutput(dataframe, top=top).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_summary_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert SummaryOutput(dataframe).get_evaluator().equal(Evaluator())


def test_summary_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert SummaryOutput(dataframe).get_evaluator(lazy=False).equal(Evaluator())
