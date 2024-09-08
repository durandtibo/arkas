from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MeanAbsoluteErrorEvaluator
from arkas.result import EmptyResult, MeanAbsoluteErrorResult, Result

################################################
#     Tests for MeanAbsoluteErrorEvaluator     #
################################################


def test_mean_absolute_error_evaluator_repr() -> None:
    assert repr(MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanAbsoluteErrorEvaluator("
    )


def test_mean_absolute_error_evaluator_str() -> None:
    assert str(MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanAbsoluteErrorEvaluator("
    )


def test_mean_absolute_error_evaluator_evaluate() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [5, 4, 3, 2, 1]}))
        .equal(
            MeanAbsoluteErrorResult(
                y_true=np.array([5, 4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4, 5])
            )
        )
    )


def test_mean_absolute_error_evaluator_evaluate_lazy_false() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(Result(metrics={"count": 5, "mean_absolute_error": 0.0}))
    )


def test_mean_absolute_error_evaluator_evaluate_missing_keys() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}))
        .equal(EmptyResult())
    )


def test_mean_absolute_error_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(EmptyResult())
    )


def test_mean_absolute_error_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
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
            MeanAbsoluteErrorResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_mean_absolute_error_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            MeanAbsoluteErrorResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )
