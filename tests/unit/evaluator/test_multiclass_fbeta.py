from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassFbetaScoreEvaluator
from arkas.result import EmptyResult, MulticlassFbetaScoreResult, Result

###################################################
#     Tests for MulticlassFbetaScoreEvaluator     #
###################################################


def test_multiclass_fbeta_score_evaluator_repr() -> None:
    assert repr(MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassFbetaScoreEvaluator("
    )


def test_multiclass_fbeta_score_evaluator_str() -> None:
    assert str(MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassFbetaScoreEvaluator("
    )


def test_multiclass_fbeta_score_evaluator_evaluate() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassFbetaScoreResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )


def test_multiclass_fbeta_score_evaluator_evaluate_betas() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassFbetaScoreResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]),
                y_pred=np.array([0, 0, 1, 1, 2, 2]),
                betas=[0.5, 1, 2],
            )
        )
    )


def test_multiclass_fbeta_score_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 1, 2, 2]}),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "f1": np.array([1.0, 1.0, 1.0]),
                    "count": 6,
                    "macro_f1": 1.0,
                    "micro_f1": 1.0,
                    "weighted_f1": 1.0,
                }
            )
        )
    )


def test_multiclass_fbeta_score_evaluator_evaluate_lazy_false_betas() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 1, 2, 2]}),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "count": 6,
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


def test_multiclass_fbeta_score_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="prediction")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(EmptyResult())
    )


def test_multiclass_fbeta_score_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassFbetaScoreEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}),
            lazy=False,
        )
        .equal(EmptyResult())
    )
