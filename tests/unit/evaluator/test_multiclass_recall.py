from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
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
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 1, 2, 2]}),
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
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(EmptyResult())
    )


def test_multiclass_recall_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multiclass_recall_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred")
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
            MulticlassRecallResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_multiclass_recall_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            MulticlassRecallResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_multiclass_recall_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        MulticlassRecallEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
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
            MulticlassRecallResult(
                y_true=np.array([1, 2, 3, 2, 1]),
                y_pred=np.array([3, 2, 0, 1, 0]),
                nan_policy=nan_policy,
            ),
        )
    )
