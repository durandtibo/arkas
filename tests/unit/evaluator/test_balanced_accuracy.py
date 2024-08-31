from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import BalancedAccuracyEvaluator
from arkas.result import BalancedAccuracyResult, EmptyResult, Result

###############################################
#     Tests for BalancedAccuracyEvaluator     #
###############################################


def test_balanced_accuracy_evaluator_repr() -> None:
    assert repr(BalancedAccuracyEvaluator(y_true="target", y_pred="pred")).startswith(
        "BalancedAccuracyEvaluator("
    )


def test_balanced_accuracy_evaluator_str() -> None:
    assert str(BalancedAccuracyEvaluator(y_true="target", y_pred="pred")).startswith(
        "BalancedAccuracyEvaluator("
    )


def test_balanced_accuracy_evaluator_evaluate() -> None:
    assert (
        BalancedAccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
        .equal(
            BalancedAccuracyResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_balanced_accuracy_evaluator_evaluate_lazy_false() -> None:
    assert (
        BalancedAccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}, lazy=False
        )
        .equal(Result(metrics={"balanced_accuracy": 1.0, "count": 5}))
    )


def test_balanced_accuracy_evaluator_evaluate_missing_keys() -> None:
    assert (
        BalancedAccuracyEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
        .equal(EmptyResult())
    )


def test_balanced_accuracy_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BalancedAccuracyEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_balanced_accuracy_evaluator_evaluate_dataframe() -> None:
    assert (
        BalancedAccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(
            BalancedAccuracyResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )
