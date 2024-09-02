from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassJaccardEvaluator
from arkas.result import EmptyResult, MulticlassJaccardResult, Result

################################################
#     Tests for MulticlassJaccardEvaluator     #
################################################


def test_multiclass_jaccard_evaluator_repr() -> None:
    assert repr(MulticlassJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassJaccardEvaluator("
    )


def test_multiclass_jaccard_evaluator_str() -> None:
    assert str(MulticlassJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassJaccardEvaluator("
    )


def test_multiclass_jaccard_evaluator_evaluate() -> None:
    assert (
        MulticlassJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate({"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 2, 2, 1])})
        .equal(
            MulticlassJaccardResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )


def test_multiclass_jaccard_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])},
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "jaccard": np.array([1.0, 1.0, 1.0]),
                    "count": 6,
                    "macro_jaccard": 1.0,
                    "micro_jaccard": 1.0,
                    "weighted_jaccard": 1.0,
                }
            )
        )
    )


def test_multiclass_jaccard_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassJaccardEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])})
        .equal(EmptyResult())
    )


def test_multiclass_jaccard_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassJaccardEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])},
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multiclass_jaccard_evaluator_evaluate_dataframe() -> None:
    assert (
        MulticlassJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassJaccardResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )
