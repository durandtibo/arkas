from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import BinaryFbetaScoreEvaluator
from arkas.result import BinaryFbetaScoreResult, EmptyResult, Result

###############################################
#     Tests for BinaryFbetaScoreEvaluator     #
###############################################


def test_binary_fbeta_score_evaluator_repr() -> None:
    assert repr(BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryFbetaScoreEvaluator("
    )


def test_binary_fbeta_score_evaluator_str() -> None:
    assert str(BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryFbetaScoreEvaluator("
    )


def test_binary_fbeta_score_evaluator_evaluate() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryFbetaScoreResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_fbeta_score_evaluator_evaluate_betas() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryFbetaScoreResult(
                y_true=np.array([1, 0, 1, 0, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                betas=[0.5, 1, 2],
            )
        )
    )


def test_binary_fbeta_score_evaluator_evaluate_lazy_false() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
        .equal(Result({"count": 5, "f1": 1.0}))
    )


def test_binary_fbeta_score_evaluator_evaluate_lazy_false_betas() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
        .equal(Result({"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0}))
    )


def test_binary_fbeta_score_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="prediction")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(EmptyResult())
    )


def test_binary_fbeta_score_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
        .equal(EmptyResult())
    )


def test_binary_fbeta_score_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryFbetaScoreResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_fbeta_score_evaluator_evaluate_drop_nulls() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred")
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
            BinaryFbetaScoreResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_binary_fbeta_score_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            BinaryFbetaScoreResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )
