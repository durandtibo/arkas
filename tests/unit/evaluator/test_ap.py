from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import AveragePrecisionEvaluator
from arkas.result import AveragePrecisionResult, EmptyResult, Result

###############################################
#     Tests for AveragePrecisionEvaluator     #
###############################################


def test_average_precision_evaluator_repr() -> None:
    assert repr(AveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "AveragePrecisionEvaluator("
    )


def test_average_precision_evaluator_str() -> None:
    assert str(AveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "AveragePrecisionEvaluator("
    )


def test_average_precision_evaluator_evaluate() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
            )
        )
    )


def test_average_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
        )
        .equal(Result(metrics={"average_precision": 1.0, "count": 5}))
    )


def test_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(EmptyResult())
    )


def test_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate(
            {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_average_precision_evaluator_evaluate_dataframe() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
            )
        )
    )
