from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MultilabelConfusionMatrixEvaluator
from arkas.result import EmptyResult, MultilabelConfusionMatrixResult, Result

########################################################
#     Tests for MultilabelConfusionMatrixEvaluator     #
########################################################


def test_multilabel_confusion_matrix_evaluator_repr() -> None:
    assert repr(MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelConfusionMatrixEvaluator("
    )


def test_multilabel_confusion_matrix_evaluator_str() -> None:
    assert str(MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "MultilabelConfusionMatrixEvaluator("
    )


def test_multilabel_confusion_matrix_evaluator_evaluate() -> None:
    assert (
        MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="pred")
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
            MultilabelConfusionMatrixResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            )
        )
    )


def test_multilabel_confusion_matrix_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            ),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "confusion_matrix": np.array(
                        [[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]
                    ),
                    "count": 5,
                },
            )
        )
    )


def test_multilabel_confusion_matrix_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="prediction")
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


def test_multilabel_confusion_matrix_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelConfusionMatrixEvaluator(y_true="target", y_pred="missing")
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
