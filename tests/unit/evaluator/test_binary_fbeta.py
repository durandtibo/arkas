from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import BinaryFbetaEvaluator
from arkas.result import BinaryFbetaResult, EmptyResult, Result

##########################################
#     Tests for BinaryFbetaEvaluator     #
##########################################


def test_binary_fbeta_evaluator_repr() -> None:
    assert repr(BinaryFbetaEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryFbetaEvaluator("
    )


def test_binary_fbeta_evaluator_str() -> None:
    assert str(BinaryFbetaEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryFbetaEvaluator("
    )


def test_binary_fbeta_evaluator_evaluate() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(
            BinaryFbetaResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )


def test_binary_fbeta_evaluator_evaluate_betas() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(
            BinaryFbetaResult(
                y_true=np.array([1, 0, 1, 0, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                betas=[0.5, 1, 2],
            )
        )
    )


def test_binary_fbeta_evaluator_evaluate_lazy_false() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([1, 0, 1, 0, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(Result({"count": 5, "f1": 1.0}))
    )


def test_binary_fbeta_evaluator_evaluate_lazy_false_betas() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(
            {"pred": np.array([1, 0, 1, 0, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(Result({"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0}))
    )


def test_binary_fbeta_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(EmptyResult())
    )


def test_binary_fbeta_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_binary_fbeta_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryFbetaEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryFbetaResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )
