from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ColumnCorrelationContentGenerator, ContentGenerator
from arkas.evaluator2 import ColumnCorrelationEvaluator, Evaluator
from arkas.output import ColumnCorrelationOutput, Output
from arkas.state import TargetDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
    )


#############################################
#     Tests for ColumnCorrelationOutput     #
#############################################


def test_column_correlation_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationOutput(")


def test_column_correlation_output_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationOutput(")


def test_column_correlation_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3")).compute(),
        Output,
    )


def test_column_correlation_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3")).equal(
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
    )


def test_column_correlation_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3")).equal(
        TargetDataFrameState(dataframe, target_column="col1")
    )


def test_column_correlation_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3")).equal(
        42
    )


def test_column_correlation_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
        .get_content_generator()
        .equal(
            ColumnCorrelationContentGenerator(
                ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
            )
        )
    )


def test_column_correlation_output_get_content_generator_lazy_false(
    dataframe: pl.DataFrame,
) -> None:
    assert isinstance(
        ColumnCorrelationOutput(
            TargetDataFrameState(dataframe, target_column="col3")
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_column_correlation_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
        .get_evaluator()
        .equal(ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3")))
    )


def test_column_correlation_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationOutput(TargetDataFrameState(dataframe, target_column="col3"))
        .get_evaluator(lazy=False)
        .equal(
            Evaluator(
                {
                    "col1": {
                        "count": 7,
                        "pearson_coeff": 1.0,
                        "pearson_pvalue": 0.0,
                        "spearman_coeff": 1.0,
                        "spearman_pvalue": 0.0,
                    },
                    "col2": {
                        "count": 7,
                        "pearson_coeff": -1.0,
                        "pearson_pvalue": 0.0,
                        "spearman_coeff": -1.0,
                        "spearman_pvalue": 0.0,
                    },
                }
            )
        )
    )
