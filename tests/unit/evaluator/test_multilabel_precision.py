from __future__ import annotations

import numpy as np

from arkas.evaluator import MultilabelPrecisionEvaluator
from arkas.result import EmptyResult, MultilabelPrecisionResult, Result

##################################################
#     Tests for MultilabelPrecisionEvaluator     #
##################################################


def test_multilabel_precision_evaluator_repr() -> None:
    assert repr(MultilabelPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelPrecisionEvaluator("
    )


def test_multilabel_precision_evaluator_str() -> None:
    assert str(MultilabelPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelPrecisionEvaluator("
    )


def test_multilabel_precision_evaluator_evaluate() -> None:
    assert (
        MultilabelPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelPrecisionResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            },
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "count": 5,
                    "macro_precision": 1.0,
                    "micro_precision": 1.0,
                    "precision": np.array([1.0, 1.0, 1.0]),
                    "weighted_precision": 1.0,
                }
            )
        )
    )


def test_multilabel_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelPrecisionEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelPrecisionEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )
