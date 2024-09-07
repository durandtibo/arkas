from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MeanAbsoluteErrorEvaluator
from arkas.result import EmptyResult, MeanAbsoluteErrorResult, Result

################################################
#     Tests for MeanAbsoluteErrorEvaluator     #
################################################


def test_mean_absolute_error_evaluator_repr() -> None:
    assert repr(MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanAbsoluteErrorEvaluator("
    )


def test_mean_absolute_error_evaluator_str() -> None:
    assert str(MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanAbsoluteErrorEvaluator("
    )


def test_mean_absolute_error_evaluator_evaluate() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([1, 2, 3, 4, 5]), "target": np.array([1, 2, 3, 4, 5])})
        .equal(
            MeanAbsoluteErrorResult(
                y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
            )
        )
    )


def test_mean_absolute_error_evaluator_evaluate_lazy_false() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([1, 2, 3, 4, 5]), "target": np.array([1, 2, 3, 4, 5])}, lazy=False
        )
        .equal(Result(metrics={"count": 5, "mean_absolute_error": 0.0}))
    )


def test_mean_absolute_error_evaluator_evaluate_missing_keys() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([1, 2, 3, 4, 5]), "target": np.array([1, 2, 3, 4, 5])})
        .equal(EmptyResult())
    )


def test_mean_absolute_error_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([1, 2, 3, 4, 5]), "target": np.array([1, 2, 3, 4, 5])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_mean_absolute_error_evaluator_evaluate_dataframe() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}))
        .equal(
            MeanAbsoluteErrorResult(
                y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
            )
        )
    )
