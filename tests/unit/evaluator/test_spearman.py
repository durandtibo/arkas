from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import SpearmanCorrelationEvaluator
from arkas.result import EmptyResult, Result, SpearmanCorrelationResult

##################################################
#     Tests for SpearmanCorrelationEvaluator     #
##################################################


def test_spearman_correlation_evaluator_repr() -> None:
    assert repr(SpearmanCorrelationEvaluator(x="target", y="pred")).startswith(
        "SpearmanCorrelationEvaluator("
    )


def test_spearman_correlation_evaluator_str() -> None:
    assert str(SpearmanCorrelationEvaluator(x="target", y="pred")).startswith(
        "SpearmanCorrelationEvaluator("
    )


def test_spearman_correlation_evaluator_evaluate() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [5, 4, 3, 2, 1]}))
        .equal(SpearmanCorrelationResult(x=np.array([5, 4, 3, 2, 1]), y=np.array([1, 2, 3, 4, 5])))
    )


def test_spearman_correlation_evaluator_evaluate_lazy_false() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="pred")
        .evaluate(
            pl.DataFrame({"pred": [1, 2, 3, 4, 5, 6], "target": [1, 2, 3, 4, 5, 6]}), lazy=False
        )
        .equal(Result(metrics={"count": 6, "spearman_coeff": 1.0, "spearman_pvalue": 0.0}))
    )


def test_spearman_correlation_evaluator_evaluate_missing_keys() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}))
        .equal(EmptyResult())
    )


def test_spearman_correlation_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(EmptyResult())
    )


def test_spearman_correlation_evaluator_evaluate_drop_nulls() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(SpearmanCorrelationResult(x=np.array([1, 2, 3, 2, 1]), y=np.array([3, 2, 0, 1, 0])))
    )


def test_spearman_correlation_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            SpearmanCorrelationResult(
                x=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_spearman_correlation_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        SpearmanCorrelationEvaluator(x="target", y="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, None],
                }
            )
        )
        .equal(
            SpearmanCorrelationResult(
                x=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                y=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                nan_policy=nan_policy,
            ),
        )
    )
