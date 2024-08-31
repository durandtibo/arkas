from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from arkas.evaluator import MultilabelRecallEvaluator
from arkas.result import EmptyResult, MultilabelRecallResult, Result

###############################################
#     Tests for MultilabelRecallEvaluator     #
###############################################


def test_multilabel_recall_evaluator_repr() -> None:
    assert repr(MultilabelRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelRecallEvaluator("
    )


def test_multilabel_recall_evaluator_str() -> None:
    assert str(MultilabelRecallEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelRecallEvaluator("
    )


def test_multilabel_recall_evaluator_evaluate() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelRecallResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_recall_evaluator_evaluate_lazy_false() -> None:
    result = MultilabelRecallEvaluator(y_true="target", y_pred="pred").evaluate(
        {
            "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            "target": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        },
        lazy=False,
    )
    assert isinstance(result, Result)
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )
    assert result.generate_figures() == {}


def test_multilabel_recall_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_recall_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )
