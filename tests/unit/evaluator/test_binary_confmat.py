from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import BinaryConfusionMatrixEvaluator
from arkas.result import BinaryConfusionMatrixResult, EmptyResult, Result

####################################################
#     Tests for BinaryConfusionMatrixEvaluator     #
####################################################


def test_binary_confusion_matrix_evaluator_repr() -> None:
    assert repr(BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryConfusionMatrixEvaluator("
    )


def test_binary_confusion_matrix_evaluator_str() -> None:
    assert str(BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryConfusionMatrixEvaluator("
    )


def test_binary_confusion_matrix_evaluator_evaluate() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryConfusionMatrixResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_confusion_matrix_evaluator_evaluate_lazy_false() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
        .equal(
            Result(
                metrics={
                    "confusion_matrix": np.array([[2, 0], [0, 3]]),
                    "count": 5,
                    "false_negative_rate": 0.0,
                    "false_negative": 0,
                    "false_positive_rate": 0.0,
                    "false_positive": 0,
                    "true_negative_rate": 1.0,
                    "true_negative": 2,
                    "true_positive_rate": 1.0,
                    "true_positive": 3,
                }
            )
        )
    )


def test_binary_confusion_matrix_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="prediction")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(EmptyResult())
    )


def test_binary_confusion_matrix_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
        .equal(EmptyResult())
    )


def test_binary_confusion_matrix_evaluator_evaluate_drop_nulls() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1, 0, 0, 1, 1, None, 1, None],
                    "target": [1, 1, 0, 1, 0, 2, None, None],
                    "col": [1, 0, 0, 1, 1, None, 7, None],
                }
            )
        )
        .equal(
            BinaryConfusionMatrixResult(
                y_true=np.array([1, 1, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_confusion_matrix_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1, 0, 0, 1, 1, None, 1, None],
                    "target": [1, 1, 0, 1, 0, 2, None, None],
                    "col": [1, 0, 0, 1, 1, None, 7, None],
                }
            )
        )
        .equal(
            BinaryConfusionMatrixResult(
                y_true=np.array([1.0, 1.0, 0.0, 1.0, 0.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_binary_confusion_matrix_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, None],
                }
            )
        )
        .equal(
            BinaryConfusionMatrixResult(
                y_true=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                nan_policy=nan_policy,
            ),
        )
    )
