from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose

from arkas.evaluator2 import ColumnCorrelationEvaluator, Evaluator
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


################################################
#     Tests for ColumnCorrelationEvaluator     #
################################################


def test_column_correlation_evaluator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationEvaluator(")


def test_column_correlation_evaluator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationEvaluator(")


def test_column_correlation_evaluator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3")).equal(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    )


def test_column_correlation_evaluator_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not ColumnCorrelationEvaluator(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col2")))


def test_column_correlation_evaluator_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ColumnCorrelationEvaluator(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(42)


def test_column_correlation_evaluator_evaluate(dataframe: pl.DataFrame) -> None:
    evaluator = ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "correlation_col1": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "correlation_col2": {
                "count": 7,
                "pearson_coeff": -1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
        },
    )


def test_column_correlation_evaluator_evaluate_drop_null_nan() -> None:
    evaluator = ColumnCorrelationEvaluator(
        TargetDataFrameState(
            pl.DataFrame(
                {
                    "col1": [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        None,
                        9.0,
                        None,
                        float("nan"),
                        12.0,
                        float("nan"),
                    ],
                    "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                    "col3": [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        None,
                        None,
                        11.0,
                        float("nan"),
                        float("nan"),
                    ],
                },
                schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
            ),
            target_column="col3",
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "correlation_col1": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "correlation_col2": {
                "count": 9,
                "pearson_coeff": -1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
        },
    )


def test_column_correlation_evaluator_evaluate_empty() -> None:
    evaluator = ColumnCorrelationEvaluator(
        TargetDataFrameState(
            pl.DataFrame({"col1": [], "col2": [], "col3": []}), target_column="col3"
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "correlation_col1": {
                "count": 0,
                "pearson_coeff": float("nan"),
                "pearson_pvalue": float("nan"),
                "spearman_coeff": float("nan"),
                "spearman_pvalue": float("nan"),
            },
            "correlation_col2": {
                "count": 0,
                "pearson_coeff": float("nan"),
                "pearson_pvalue": float("nan"),
                "spearman_coeff": float("nan"),
                "spearman_pvalue": float("nan"),
            },
        },
        equal_nan=True,
    )


def test_column_correlation_evaluator_evaluate_prefix_suffix(dataframe: pl.DataFrame) -> None:
    evaluator = ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    assert objects_are_allclose(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_correlation_col1_suffix": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "prefix_correlation_col2_suffix": {
                "count": 7,
                "pearson_coeff": -1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
        },
    )


def test_column_correlation_evaluator_compute(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        .compute()
        .equal(
            Evaluator(
                {
                    "correlation_col1": {
                        "count": 7,
                        "pearson_coeff": 1.0,
                        "pearson_pvalue": 0.0,
                        "spearman_coeff": 1.0,
                        "spearman_pvalue": 0.0,
                    },
                    "correlation_col2": {
                        "count": 7,
                        "pearson_coeff": -1.0,
                        "pearson_pvalue": 0.0,
                        "spearman_coeff": -1.0,
                        "spearman_pvalue": 0.0,
                    },
                },
            )
        )
    )
