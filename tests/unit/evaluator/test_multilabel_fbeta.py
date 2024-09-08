from __future__ import annotations

import numpy as np

from arkas.evaluator import MultilabelFbetaScoreEvaluator
from arkas.result import EmptyResult, MultilabelFbetaScoreResult, Result

###################################################
#     Tests for MultilabelFbetaScoreEvaluator     #
###################################################


def test_multilabel_fbeta_score_evaluator_repr() -> None:
    assert repr(MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelFbetaScoreEvaluator("
    )


def test_multilabel_fbeta_score_evaluator_str() -> None:
    assert str(MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelFbetaScoreEvaluator("
    )


def test_multilabel_fbeta_score_evaluator_evaluate() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelFbetaScoreResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_fbeta_score_evaluator_evaluate_betas() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelFbetaScoreResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                betas=[0.5, 1, 2],
            )
        )
    )


def test_multilabel_fbeta_score_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred")
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


def test_multilabel_fbeta_score_evaluator_evaluate_lazy_false_beteas() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
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


def test_multilabel_fbeta_score_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="prediction")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_fbeta_score_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {
                "pred": np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )
