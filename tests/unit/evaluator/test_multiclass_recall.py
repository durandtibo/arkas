from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassRecallEvaluator
from arkas.result import EmptyResult, MulticlassRecallResult, Result

###############################################
#     Tests for MulticlassRecallEvaluator     #
###############################################


def test_multiclass_recall_evaluator_repr() -> None:
    assert repr(MulticlassRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassRecallEvaluator("
    )


def test_multiclass_recall_evaluator_str() -> None:
    assert str(MulticlassRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassRecallEvaluator("
    )


def test_multiclass_recall_evaluator_evaluate() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 2, 2, 1])})
        .equal(
            MulticlassRecallResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )


def test_multiclass_recall_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])},
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "recall": np.array([1.0, 1.0, 1.0]),
                    "count": 6,
                    "macro_recall": 1.0,
                    "micro_recall": 1.0,
                    "weighted_recall": 1.0,
                }
            )
        )
    )


def test_multiclass_recall_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])})
        .equal(EmptyResult())
    )


def test_multiclass_recall_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])},
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multiclass_recall_evaluator_evaluate_dataframe() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassRecallResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )
