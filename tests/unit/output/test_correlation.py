from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, CorrelationContentGenerator
from arkas.evaluator2 import CorrelationEvaluator, Evaluator
from arkas.output import CorrelationOutput, Output
from arkas.state import TwoColumnDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64},
    )


#######################################
#     Tests for CorrelationOutput     #
#######################################


def test_correlation_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationOutput(")


def test_correlation_output_str(dataframe: pl.DataFrame) -> None:
    assert str(
        CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationOutput(")


def test_correlation_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationOutput(
            TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
        ).compute(),
        Output,
    )


def test_correlation_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert CorrelationOutput(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")))


def test_correlation_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not CorrelationOutput(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(
        TwoColumnDataFrameState(
            pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
        )
    )


def test_correlation_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not CorrelationOutput(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(42)


def test_correlation_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
        .get_content_generator()
        .equal(
            CorrelationContentGenerator(
                TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
            )
        )
    )


def test_correlation_output_get_content_generator_lazy_false(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        CorrelationOutput(
            TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_correlation_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
        .get_evaluator()
        .equal(
            CorrelationEvaluator(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
        )
    )


def test_correlation_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        CorrelationOutput(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
        .get_evaluator(lazy=False)
        .equal(
            Evaluator(
                {
                    "count": 7,
                    "pearson_coeff": -1.0,
                    "pearson_pvalue": 0.0,
                    "spearman_coeff": -1.0,
                    "spearman_pvalue": 0.0,
                }
            )
        )
    )
