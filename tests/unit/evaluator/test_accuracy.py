from __future__ import annotations

import numpy as np
import polars as pl

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
        .evaluate({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
        .equal(AccuracyResult(y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])))
    )


def test_accuracy_evaluator_evaluate_lazy_false() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}, lazy=False
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
        .evaluate({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
        .equal(EmptyResult())
    )


def test_accuracy_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_accuracy_evaluator_evaluate_dataframe() -> None:
    assert (
        AccuracyEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(AccuracyResult(y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])))
    )
