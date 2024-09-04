from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import BinaryRocAucEvaluator
from arkas.result import BinaryRocAucResult, EmptyResult, Result

###########################################
#     Tests for BinaryRocAucEvaluator     #
###########################################


def test_binary_roc_auc_evaluator_repr() -> None:
    assert repr(BinaryRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "BinaryRocAucEvaluator("
    )


def test_binary_roc_auc_evaluator_str() -> None:
    assert str(BinaryRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "BinaryRocAucEvaluator("
    )


def test_binary_roc_auc_evaluator_evaluate() -> None:
    assert (
        BinaryRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(
            BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
        )
    )


def test_binary_roc_auc_evaluator_evaluate_lazy_false() -> None:
    assert (
        BinaryRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
        )
        .equal(Result(metrics={"count": 5, "roc_auc": 1.0}))
    )


def test_binary_roc_auc_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryRocAucEvaluator(y_true="target", y_score="prediction")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(EmptyResult())
    )


def test_binary_roc_auc_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryRocAucEvaluator(y_true="target", y_score="missing")
        .evaluate(
            {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_binary_roc_auc_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryRocAucResult(y_true=np.array([1, 0, 1, 0, 1]), y_score=np.array([2, -1, 0, 3, 1]))
        )
    )
