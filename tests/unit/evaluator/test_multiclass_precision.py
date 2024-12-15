from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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


def test_multiclass_precision_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            MulticlassPrecisionResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_multiclass_precision_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            MulticlassPrecisionResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_multiclass_precision_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        MulticlassPrecisionEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            MulticlassPrecisionResult(
                y_true=np.array([1, 2, 3, 2, 1]),
                y_pred=np.array([3, 2, 0, 1, 0]),
                nan_policy=nan_policy,
            ),
        )
    )
