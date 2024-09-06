from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal
from matplotlib import pyplot as plt

from arkas.evaluator import BinaryRecallEvaluator
from arkas.result import BinaryRecallResult, EmptyResult, Result

###########################################
#     Tests for BinaryRecallEvaluator     #
###########################################


def test_binary_recall_evaluator_repr() -> None:
    assert repr(BinaryRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryRecallEvaluator("
    )


def test_binary_recall_evaluator_str() -> None:
    assert str(BinaryRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryRecallEvaluator("
    )


def test_binary_recall_evaluator_evaluate() -> None:
    assert (
        BinaryRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(
            BinaryRecallResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )


def test_binary_recall_evaluator_evaluate_lazy_false() -> None:
    result = BinaryRecallEvaluator(y_true="target", y_pred="pred").evaluate(
        {"pred": np.array([1, 0, 1, 0, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
    )
    assert isinstance(result, Result)
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "recall": 1.0})
    figures = result.generate_figures()
    assert len(figures) == 1
    assert isinstance(figures["precision_recall"], plt.Figure)


def test_binary_recall_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryRecallEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(EmptyResult())
    )


def test_binary_recall_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryRecallEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_binary_recall_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryRecallResult(y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
        )
    )
