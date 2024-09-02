from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MultilabelAveragePrecisionEvaluator
from arkas.result import EmptyResult, MultilabelAveragePrecisionResult, Result

#########################################################
#     Tests for MultilabelAveragePrecisionEvaluator     #
#########################################################


def test_multilabel_average_precision_evaluator_repr() -> None:
    assert repr(MultilabelAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "MultilabelAveragePrecisionEvaluator("
    )


def test_multilabel_average_precision_evaluator_str() -> None:
    assert str(MultilabelAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "MultilabelAveragePrecisionEvaluator("
    )


def test_multilabel_average_precision_evaluator_evaluate() -> None:
    assert (
        MultilabelAveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {
                "pred": np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(
            MultilabelAveragePrecisionResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )


def test_multilabel_average_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        MultilabelAveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {
                "pred": np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "average_precision": np.array([1.0, 1.0, 1.0]),
                    "count": 5,
                    "macro_average_precision": 1.0,
                    "micro_average_precision": 1.0,
                    "weighted_average_precision": 1.0,
                }
            )
        )
    )


def test_multilabel_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        MultilabelAveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate(
            {
                "pred": np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            }
        )
        .equal(EmptyResult())
    )


def test_multilabel_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MultilabelAveragePrecisionEvaluator(y_true="target", y_score="missing")
        .evaluate(
            {
                "pred": np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multilabel_average_precision_evaluator_evaluate_dataframe() -> None:
    assert (
        MultilabelAveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
                    "target": np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                }
            )
        )
        .equal(
            MultilabelAveragePrecisionResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            )
        )
    )
