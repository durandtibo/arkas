from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose

from arkas.evaluator2 import CorrelationEvaluator, Evaluator
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


##########################################
#     Tests for CorrelationEvaluator     #
##########################################


def test_correlation_evaluator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        CorrelationEvaluator(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationEvaluator(")


def test_correlation_evaluator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        CorrelationEvaluator(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    ).startswith("CorrelationEvaluator(")


def test_correlation_evaluator_state(dataframe: pl.DataFrame) -> None:
    assert CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).state.equal(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))


def test_correlation_evaluator_equal_true(dataframe: pl.DataFrame) -> None:
    assert CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(
        CorrelationEvaluator(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
    )


def test_correlation_evaluator_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(
        CorrelationEvaluator(
            TwoColumnDataFrameState(
                pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
            )
        )
    )


def test_correlation_evaluator_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    ).equal(42)


def test_correlation_evaluator_evaluate(dataframe: pl.DataFrame) -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 7,
            "pearson_coeff": -1.0,
            "pearson_pvalue": 0.0,
            "spearman_coeff": -1.0,
            "spearman_pvalue": 0.0,
        },
    )


def test_correlation_evaluator_evaluate_one_row() -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(
            pl.DataFrame(
                {"col1": [1.0], "col2": [7.0]}, schema={"col1": pl.Float64, "col2": pl.Float64}
            ),
            column1="col1",
            column2="col2",
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 1,
            "pearson_coeff": float("nan"),
            "pearson_pvalue": float("nan"),
            "spearman_coeff": float("nan"),
            "spearman_pvalue": float("nan"),
        },
        equal_nan=True,
    )


def test_correlation_evaluator_evaluate_nan_policy_omit() -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(
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
                    "col2": [
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
                schema={"col1": pl.Float64, "col2": pl.Float64},
            ),
            column1="col1",
            column2="col2",
            nan_policy="omit",
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 7,
            "pearson_coeff": 1.0,
            "pearson_pvalue": 0.0,
            "spearman_coeff": 1.0,
            "spearman_pvalue": 0.0,
        },
    )


def test_correlation_evaluator_evaluate_nan_policy_propagate() -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(
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
                    "col2": [
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
                schema={"col1": pl.Float64, "col2": pl.Float64},
            ),
            column1="col1",
            column2="col2",
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 13,
            "pearson_coeff": float("nan"),
            "pearson_pvalue": float("nan"),
            "spearman_coeff": float("nan"),
            "spearman_pvalue": float("nan"),
        },
        equal_nan=True,
    )


def test_correlation_evaluator_evaluate_nan_policy_raise() -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(
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
                    "col2": [
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
                schema={"col1": pl.Float64, "col2": pl.Float64},
            ),
            nan_policy="raise",
            column1="col1",
            column2="col2",
        )
    )
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        evaluator.evaluate()


def test_correlation_evaluator_evaluate_empty() -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(
            pl.DataFrame({"col1": [], "col2": []}), column1="col1", column2="col2"
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 0,
            "pearson_coeff": float("nan"),
            "pearson_pvalue": float("nan"),
            "spearman_coeff": float("nan"),
            "spearman_pvalue": float("nan"),
        },
        equal_nan=True,
    )


def test_correlation_evaluator_evaluate_prefix_suffix(dataframe: pl.DataFrame) -> None:
    evaluator = CorrelationEvaluator(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    )
    assert objects_are_allclose(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_count_suffix": 7,
            "prefix_pearson_coeff_suffix": -1.0,
            "prefix_pearson_pvalue_suffix": 0.0,
            "prefix_spearman_coeff_suffix": -1.0,
            "prefix_spearman_pvalue_suffix": 0.0,
        },
    )


def test_correlation_evaluator_compute(dataframe: pl.DataFrame) -> None:
    assert (
        CorrelationEvaluator(TwoColumnDataFrameState(dataframe, column1="col1", column2="col2"))
        .compute()
        .equal(
            Evaluator(
                {
                    "count": 7,
                    "pearson_coeff": -1.0,
                    "pearson_pvalue": 0.0,
                    "spearman_coeff": -1.0,
                    "spearman_pvalue": 0.0,
                },
            )
        )
    )
