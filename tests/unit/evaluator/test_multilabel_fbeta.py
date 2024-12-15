from __future__ import annotations

import numpy as np
import polars as pl
import pytest

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
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
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
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
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
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
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


def test_multilabel_fbeta_score_evaluator_evaluate_lazy_false_betas() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2])
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
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
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
        )
        .equal(EmptyResult())
    )


def test_multilabel_fbeta_score_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multilabel_fbeta_score_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [
                        [2, -1, 1],
                        [-1, 1, -2],
                        [0, 2, -3],
                        [3, -2, 4],
                        [1, -3, 5],
                        [0, 1, 0],
                        None,
                        None,
                    ],
                    "target": [
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 1],
                        None,
                        [0, 1, 0],
                        None,
                    ],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                },
                schema={
                    "pred": pl.Array(pl.Int64, 3),
                    "target": pl.Array(pl.Int64, 3),
                    "col": pl.Int64,
                },
            )
        )
        .equal(
            MultilabelFbetaScoreResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )


def test_multilabel_fbeta_score_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [
                        [2, -1, 1],
                        [-1, 1, -2],
                        [0, 2, -3],
                        [3, -2, 4],
                        [1, -3, 5],
                        [0, 1, 0],
                        None,
                        None,
                    ],
                    "target": [
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 1],
                        None,
                        [0, 1, 0],
                        None,
                    ],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                },
                schema={
                    "pred": pl.Array(pl.Int64, 3),
                    "target": pl.Array(pl.Int64, 3),
                    "col": pl.Int64,
                },
            )
        )
        .equal(
            MultilabelFbetaScoreResult(
                y_true=np.array(
                    [
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 1],
                        [float("nan"), float("nan"), float("nan")],
                        [0, 1, 0],
                        [float("nan"), float("nan"), float("nan")],
                    ]
                ),
                y_pred=np.array(
                    [
                        [2, -1, 1],
                        [-1, 1, -2],
                        [0, 2, -3],
                        [3, -2, 4],
                        [1, -3, 5],
                        [0, 1, 0],
                        [float("nan"), float("nan"), float("nan")],
                        [float("nan"), float("nan"), float("nan")],
                    ]
                ),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_multilabel_fbeta_score_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        MultilabelFbetaScoreEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [
                        [2, -1, 1],
                        [-1, 1, -2],
                        [0, 2, -3],
                        [3, -2, 4],
                        [1, -3, 5],
                        [0, 1, 0],
                        None,
                        None,
                    ],
                    "target": [
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0],
                        [1, 0, 1],
                        [1, 0, 1],
                        None,
                        [0, 1, 0],
                        None,
                    ],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                },
                schema={
                    "pred": pl.Array(pl.Int64, 3),
                    "target": pl.Array(pl.Int64, 3),
                    "col": pl.Int64,
                },
            )
        )
        .equal(
            MultilabelFbetaScoreResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                nan_policy=nan_policy,
            ),
        )
    )
