from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import AveragePrecisionEvaluator
from arkas.result import AveragePrecisionResult, EmptyResult, Result

###############################################
#     Tests for AveragePrecisionEvaluator     #
###############################################


def test_average_precision_evaluator_repr() -> None:
    assert repr(AveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "AveragePrecisionEvaluator("
    )


def test_average_precision_evaluator_str() -> None:
    assert str(AveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "AveragePrecisionEvaluator("
    )


def test_average_precision_evaluator_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'label_type': incorrect"):
        AveragePrecisionEvaluator(y_true="target", y_score="pred", label_type="incorrect")


def test_average_precision_evaluator_evaluate() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_score=np.array([2, -1, 0, 3, 1]),
                label_type="binary",
            )
        )
    )


def test_average_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}), lazy=False)
        .equal(Result(metrics={"average_precision": 1.0, "count": 5}))
    )


def test_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(EmptyResult())
    )


def test_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}), lazy=False)
        .equal(EmptyResult())
    )


def test_average_precision_evaluator_evaluate_binary() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred", label_type="binary")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_score=np.array([2, -1, 0, 3, 1]),
                label_type="binary",
            )
        )
    )


def test_average_precision_evaluator_evaluate_multiclass() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred", label_type="multiclass")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ],
                    "target": [0, 0, 1, 1, 2, 2],
                },
                schema={"pred": pl.Array(pl.Float64, 3), "target": pl.Int64},
            )
        )
        .equal(
            AveragePrecisionResult(
                y_true=np.array([0, 0, 1, 1, 2, 2]),
                y_score=np.array(
                    [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                label_type="multiclass",
            )
        )
    )


def test_average_precision_evaluator_evaluate_multilabel() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred", label_type="multilabel")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]],
                    "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
                },
                schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
            )
        )
        .equal(
            AveragePrecisionResult(
                y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
                y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
                label_type="multilabel",
            )
        )
    )


def test_average_precision_evaluator_evaluate_drop_nulls() -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred", label_type="binary")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [2, -1, 0, 3, 1, None, 1, None],
                    "target": [1, 0, 0, 1, 1, 2, None, None],
                    "col": [1, 0, 0, 1, 1, None, 7, None],
                }
            )
        )
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_score=np.array([2, -1, 0, 3, 1]),
                label_type="binary",
            )
        )
    )


def test_average_precision_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        AveragePrecisionEvaluator(
            y_true="target", y_score="pred", label_type="binary", drop_nulls=False
        )
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [2, -1, 0, 3, 1, None, 1, None],
                    "target": [1, 0, 0, 1, 1, 2, None, None],
                    "col": [1, 0, 0, 1, 1, None, 7, None],
                }
            )
        )
        .equal(
            AveragePrecisionResult(
                y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_score=np.array([2.0, -1.0, 0.0, 3.0, 1.0, float("nan"), 1.0, float("nan")]),
                label_type="binary",
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_average_precision_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        AveragePrecisionEvaluator(y_true="target", y_score="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, None],
                }
            )
        )
        .equal(
            AveragePrecisionResult(
                y_true=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                y_score=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                nan_policy=nan_policy,
            ),
        )
    )
