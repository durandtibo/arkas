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
