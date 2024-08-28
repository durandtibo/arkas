from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

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
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(
            BinaryPrecisionResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_precision_evaluator_evaluate_lazy_false() -> None:
    result = BinaryPrecisionEvaluator(y_true="target", y_pred="pred").evaluate(
        {"pred": np.array([1, 0, 1, 0, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
    )
    assert isinstance(result, Result)
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "precision": 1.0})
    assert len(result.generate_figures()) == 1


def test_binary_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])})
        .equal(EmptyResult())
    )


def test_binary_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryPrecisionEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([1, 0, 0, 1, 1]), "target": np.array([1, 0, 1, 0, 1])}, lazy=False
        )
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
