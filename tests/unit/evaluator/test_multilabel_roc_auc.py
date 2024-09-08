from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MultilabelRocAucEvaluator
from arkas.result import EmptyResult, MultilabelRocAucResult, Result

###############################################
#     Tests for MultilabelRocAucEvaluator     #
###############################################


def test_multilabel_roc_auc_evaluator_repr() -> None:
    assert repr(MultilabelRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "MultilabelRocAucEvaluator("
    )


def test_multilabel_roc_auc_evaluator_str() -> None:
    assert str(MultilabelRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "MultilabelRocAucEvaluator("
    )


def test_multilabel_roc_auc_evaluator_evaluate() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
        )
        .equal(
            MultilabelRocAucResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )


def test_multilabel_roc_auc_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "count": 5,
                    "macro_roc_auc": 1.0,
                    "micro_roc_auc": 1.0,
                    "roc_auc": np.array([1.0, 1.0, 1.0]),
                    "weighted_roc_auc": 1.0,
                }
            )
        )
    )


def test_multilabel_roc_auc_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="prediction")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
        )
        .equal(EmptyResult())
    )


def test_multilabel_roc_auc_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="missing")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multilabel_roc_auc_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="pred")
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
            MultilabelRocAucResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )


def test_multilabel_roc_auc_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MultilabelRocAucEvaluator(y_true="target", y_score="pred", drop_nulls=False)
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
            MultilabelRocAucResult(
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
                y_score=np.array(
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
