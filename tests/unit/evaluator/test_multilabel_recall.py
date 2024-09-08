from __future__ import annotations

import numpy as np
import polars as pl

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
            pl.DataFrame(
                {
                    "pred": [[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
        )
        .equal(
            MultilabelRecallResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_recall_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="pred")
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
                    "macro_recall": 1.0,
                    "micro_recall": 1.0,
                    "recall": np.array([1.0, 1.0, 1.0]),
                    "weighted_recall": 1.0,
                }
            )
        )
    )


def test_multilabel_recall_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="prediction")
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


def test_multilabel_recall_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="missing")
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


def test_multilabel_recall_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="pred")
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
            MultilabelRecallResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )


def test_multilabel_recall_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MultilabelRecallEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            MultilabelRecallResult(
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
