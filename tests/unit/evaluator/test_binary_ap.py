from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

from arkas.evaluator import BinaryAveragePrecisionEvaluator
from arkas.result import BinaryAveragePrecisionResult, EmptyResult, Result

#####################################################
#     Tests for BinaryAveragePrecisionEvaluator     #
#####################################################


def test_binary_average_precision_evaluator_repr() -> None:
    assert repr(BinaryAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "BinaryAveragePrecisionEvaluator("
    )


def test_binary_average_precision_evaluator_str() -> None:
    assert str(BinaryAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "BinaryAveragePrecisionEvaluator("
    )


def test_binary_average_precision_evaluator_evaluate() -> None:
    assert (
        BinaryAveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(
            BinaryAveragePrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
            )
        )
    )


def test_binary_average_precision_evaluator_evaluate_lazy_false() -> None:
    result = BinaryAveragePrecisionEvaluator(y_true="target", y_score="pred").evaluate(
        {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
    )
    assert isinstance(result, Result)
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "average_precision": 1.0})
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        BinaryAveragePrecisionEvaluator(y_true="target", y_score="prediction")
        .evaluate({"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])})
        .equal(EmptyResult())
    )


def test_binary_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryAveragePrecisionEvaluator(y_true="target", y_score="missing")
        .evaluate(
            {"pred": np.array([2, -1, 0, 3, 1]), "target": np.array([1, 0, 0, 1, 1])}, lazy=False
        )
        .equal(EmptyResult())
    )


def test_binary_average_precision_evaluator_evaluate_dataframe() -> None:
    assert (
        BinaryAveragePrecisionEvaluator(y_true="target", y_score="pred")
        .evaluate(pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            BinaryAveragePrecisionResult(
                y_true=np.array([1, 0, 1, 0, 1]), y_score=np.array([2, -1, 0, 3, 1])
            )
        )
    )
