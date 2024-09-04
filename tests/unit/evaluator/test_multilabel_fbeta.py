from __future__ import annotations

import numpy as np

from arkas.evaluator import MultilabelFbetaEvaluator
from arkas.result import EmptyResult, MultilabelFbetaResult, Result

##############################################
#     Tests for MultilabelFbetaEvaluator     #
##############################################


def test_multilabel_fbeta_evaluator_repr() -> None:
    assert repr(MultilabelFbetaEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelFbetaEvaluator("
    )


def test_multilabel_fbeta_evaluator_str() -> None:
    assert str(MultilabelFbetaEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelFbetaEvaluator("
    )


def test_multilabel_fbeta_evaluator_evaluate() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelFbetaResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_fbeta_evaluator_evaluate_betas() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelFbetaResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                betas=[0.5, 1, 2],
            )
        )
    )


def test_multilabel_fbeta_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="pred")
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
                    "macro_f1": 1.0,
                    "micro_f1": 1.0,
                    "f1": np.array([1.0, 1.0, 1.0]),
                    "weighted_f1": 1.0,
                }
            )
        )
    )


def test_multilabel_fbeta_evaluator_evaluate_lazy_false_beteas() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
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
                    "f0.5": np.array([1.0, 1.0, 1.0]),
                    "macro_f0.5": 1.0,
                    "micro_f0.5": 1.0,
                    "weighted_f0.5": 1.0,
                    "f1": np.array([1.0, 1.0, 1.0]),
                    "macro_f1": 1.0,
                    "micro_f1": 1.0,
                    "weighted_f1": 1.0,
                    "f2": np.array([1.0, 1.0, 1.0]),
                    "macro_f2": 1.0,
                    "micro_f2": 1.0,
                    "weighted_f2": 1.0,
                }
            )
        )
    )


def test_multilabel_fbeta_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_fbeta_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelFbetaEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )
