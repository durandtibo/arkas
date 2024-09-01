from __future__ import annotations

import numpy as np

from arkas.evaluator import MultilabelJaccardEvaluator
from arkas.result import EmptyResult, MultilabelJaccardResult, Result

################################################
#     Tests for MultilabelJaccardEvaluator     #
################################################


def test_multilabel_jaccard_evaluator_repr() -> None:
    assert repr(MultilabelJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelJaccardEvaluator("
    )


def test_multilabel_jaccard_evaluator_str() -> None:
    assert str(MultilabelJaccardEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelJaccardEvaluator("
    )


def test_multilabel_jaccard_evaluator_evaluate() -> None:
    assert (
        MultilabelJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelJaccardResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_jaccard_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelJaccardEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            },
            lazy=False,
        )
        .equal(
            Result(
                {
                    "count": 5,
                    "macro_jaccard": 1.0,
                    "micro_jaccard": 1.0,
                    "jaccard": np.array([1.0, 1.0, 1.0]),
                    "weighted_jaccard": 1.0,
                }
            )
        )
    )


def test_multilabel_jaccard_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelJaccardEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_jaccard_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelJaccardEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )
