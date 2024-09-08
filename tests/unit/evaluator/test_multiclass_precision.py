from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassPrecisionEvaluator
from arkas.result import EmptyResult, MulticlassPrecisionResult, Result

##################################################
#     Tests for MulticlassPrecisionEvaluator     #
##################################################


def test_multiclass_precision_evaluator_repr() -> None:
    assert repr(MulticlassPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassPrecisionEvaluator("
    )


def test_multiclass_precision_evaluator_str() -> None:
    assert str(MulticlassPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassPrecisionEvaluator("
    )


def test_multiclass_precision_evaluator_evaluate() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassPrecisionResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )


def test_multiclass_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 1, 2, 2]}),
            lazy=False,
        )
        .equal(
            Result(
                {
                    "precision": np.array([1.0, 1.0, 1.0]),
                    "count": 6,
                    "macro_precision": 1.0,
                    "micro_precision": 1.0,
                    "weighted_precision": 1.0,
                }
            )
        )
    )


def test_multiclass_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="prediction")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(EmptyResult())
    )


def test_multiclass_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}),
            lazy=False,
        )
        .equal(EmptyResult())
    )
