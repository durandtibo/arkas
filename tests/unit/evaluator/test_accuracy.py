from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import AccuracyEvaluator
from arkas.result import AccuracyResult, EmptyResult, Result

#######################################
#     Tests for AccuracyEvaluator     #
#######################################


def test_accuracy_evaluator_repr() -> None:
    assert repr(AccuracyEvaluator(y_true="target", y_pred="pred")).startswith("AccuracyEvaluator(")


def test_accuracy_evaluator_str() -> None:
    assert str(AccuracyEvaluator(y_true="target", y_pred="pred")).startswith("AccuracyEvaluator(")


def test_accuracy_evaluator_evaluate() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(AccuracyResult(y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])))
    )


def test_accuracy_evaluator_evaluate_lazy_false() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [3, 2, 0, 1, 0]}),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "accuracy": 1.0,
                    "count": 5,
                    "count_correct": 5,
                    "count_incorrect": 0,
                    "error": 0.0,
                }
            )
        )
    )


def test_accuracy_evaluator_evaluate_missing_keys() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
        )
        .equal(EmptyResult())
    )


def test_accuracy_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_accuracy_evaluator_evaluate_drop_nulls() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(AccuracyResult(y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])))
    )


def test_accuracy_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            AccuracyResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, float("nan")],
                }
            )
        )
        .equal(
            AccuracyResult(
                y_true=np.array([5.0, 4.0, 3.0, 2.0, 1.0, float("nan")]),
                y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]),
                nan_policy=nan_policy,
            ),
            equal_nan=True,
        )
    )
