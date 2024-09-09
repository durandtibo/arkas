from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal
from matplotlib import pyplot as plt

from arkas.evaluator import BinaryClassificationEvaluator
from arkas.result import BinaryClassificationResult, EmptyResult, Result

###################################################
#     Tests for BinaryClassificationEvaluator     #
###################################################


def test_binary_classification_evaluator_repr() -> None:
    assert repr(BinaryClassificationEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryClassificationEvaluator("
    )


def test_binary_classification_evaluator_str() -> None:
    assert str(BinaryClassificationEvaluator(y_true="target", y_pred="pred")).startswith(
        "BinaryClassificationEvaluator("
    )


def test_binary_classification_evaluator_evaluate() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(
            BinaryClassificationResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
            )
        )
    )


def test_binary_classification_evaluator_evaluate_with_score() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="pred", y_score="score")
        .evaluate(
            pl.DataFrame(
                {"pred": [1, 0, 0, 1, 1], "score": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}
            )
        )
        .equal(
            BinaryClassificationResult(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_score=np.array([2, -1, 0, 3, 1]),
            )
        )
    )


def test_binary_classification_evaluator_evaluate_lazy_false() -> None:
    result = BinaryClassificationEvaluator(
        y_true="target", y_pred="pred", y_score="score"
    ).evaluate(
        pl.DataFrame(
            {
                "pred": [1, 0, 0, 1, 1],
                "score": [2, -1, 0, 3, 1],
                "target": [1, 0, 0, 1, 1],
            }
        ),
        lazy=False,
    )
    assert isinstance(result, Result)
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": 1.0,
            "count_correct": 5,
            "count_incorrect": 0,
            "count": 5,
            "error": 0.0,
            "balanced_accuracy": 1.0,
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
        },
    )
    figures = result.generate_figures()
    assert len(figures) == 1
    assert isinstance(figures["precision_recall"], plt.Figure)


def test_binary_classification_evaluator_evaluate_missing_keys_pred() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame(
                {"pred": [1, 0, 0, 1, 1], "score": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}
            )
        )
        .equal(EmptyResult())
    )


def test_binary_classification_evaluator_evaluate_missing_keys_score() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="pred", y_score="score")
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 0, 1, 1]}))
        .equal(EmptyResult())
    )


def test_binary_classification_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            pl.DataFrame(
                {"pred": [1, 0, 0, 1, 1], "score": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]}
            ),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_binary_classification_evaluator_evaluate_drop_nulls() -> None:
    assert (
        BinaryClassificationEvaluator(y_true="target", y_pred="pred", y_score="score")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, 2, None],
                    "score": [2, -1, 0, 3, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, 1, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None, 9],
                }
            )
        )
        .equal(
            BinaryClassificationResult(
                y_true=np.array([1, 2, 3, 2, 1]),
                y_pred=np.array([3, 2, 0, 1, 0]),
                y_score=np.array([2, -1, 0, 3, 1]),
            )
        )
    )


def test_binary_classification_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        BinaryClassificationEvaluator(
            y_true="target", y_pred="pred", y_score="score", drop_nulls=False
        )
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, 2, None],
                    "score": [2, -1, 0, 3, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, 1, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None, 9],
                }
            )
        )
        .equal(
            BinaryClassificationResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, 2.0, float("nan")]),
                y_score=np.array([2.0, -1.0, 0.0, 3.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )
