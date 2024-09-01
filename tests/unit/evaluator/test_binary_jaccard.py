from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import BinaryJaccardEvaluator
from arkas.result import BinaryJaccardResult, EmptyResult, Result

############################################
#     Tests for BinaryJaccardEvaluator     #
############################################


def test_binary_jaccard_evaluator_repr() -> None:
    assert repr(BinaryJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryJaccardEvaluator("
    )


def test_binary_jaccard_evaluator_str() -> None:
    assert str(BinaryJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryJaccardEvaluator("
    )


def test_binary_jaccard_evaluator_evaluate() -> None:
    assert (
        BinaryJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(
            BinaryJaccardResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )


def test_binary_jaccard_evaluator_evaluate_lazy_false() -> None:
    assert (
        BinaryJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([1, 0, 1, 0, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(Result(metrics={"count": 5, "jaccard": 1.0}))
    )


def test_binary_jaccard_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryJaccardEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(EmptyResult())
    )


def test_binary_jaccard_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryJaccardEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_binary_jaccard_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryJaccardResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )
