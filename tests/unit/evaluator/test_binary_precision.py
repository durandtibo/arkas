from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from matplotlib import pyplot as plt

from arkas.evaluator import BinaryPrecisionEvaluator
from arkas.result import BinaryPrecisionResult, EmptyResult, Result

##############################################
#     Tests for BinaryPrecisionEvaluator     #
##############################################


def test_binary_precision_evaluator_repr() -> None:
    assert repr(BinaryPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryPrecisionEvaluator("
    )


def test_binary_precision_evaluator_str() -> None:
    assert str(BinaryPrecisionEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryPrecisionEvaluator("
    )


def test_binary_precision_evaluator_evaluate() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryPrecisionResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_precision_evaluator_evaluate_lazy_false() -> None:
    result = BinaryPrecisionEvaluator(y_true="target", y_pred="pred").evaluate(
        pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False
    )
    assert isinstance(result, Result)
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "precision": 1.0})
    figures = result.generate_figures()
    assert len(figures) == 1
    assert isinstance(figures["precision_recall"], plt.Figure)


def test_binary_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="prediction")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(EmptyResult())
    )


def test_binary_precision_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryPrecisionResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_precision_evaluator_evaluate_drop_nulls() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="pred")
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
            BinaryPrecisionResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_binary_precision_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
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
            BinaryPrecisionResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_binary_precision_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, None],
                }
            )
        )
        .equal(
            BinaryPrecisionResult(
                y_true=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                nan_policy=nan_policy,
            ),
        )
    )
