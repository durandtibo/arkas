from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassConfusionMatrixEvaluator
from arkas.result import EmptyResult, MulticlassConfusionMatrixResult, Result

########################################################
#     Tests for MulticlassConfusionMatrixEvaluator     #
########################################################


def test_multiclass_confusion_matrix_evaluator_repr() -> None:
    assert repr(MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassConfusionMatrixEvaluator("
    )


def test_multiclass_confusion_matrix_evaluator_str() -> None:
    assert str(MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="pred")).startswith(
        "MulticlassConfusionMatrixEvaluator("
    )


def test_multiclass_confusion_matrix_evaluator_evaluate() -> None:
    assert (
        MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 2, 2, 1]}))
        .equal(
            MulticlassConfusionMatrixResult(
                y_true=np.array([0, 0, 1, 2, 2, 1]), y_pred=np.array([0, 0, 1, 1, 2, 2])
            )
        )
    )


def test_multiclass_confusion_matrix_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame({"pred": [0, 1, 1, 2, 2, 2], "target": [0, 1, 1, 2, 2, 2]}),
            lazy=False,
        )
        .equal(
            Result(
                {
                    "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
                    "count": 6,
                }
            )
        )
    )


def test_multiclass_confusion_matrix_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="prediction")
        .evaluate({"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])})
        .equal(EmptyResult())
    )


def test_multiclass_confusion_matrix_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassConfusionMatrixEvaluator(y_true="target", y_pred="missing")
        .evaluate(
            {"pred": np.array([0, 0, 1, 1, 2, 2]), "target": np.array([0, 0, 1, 1, 2, 2])},
            lazy=False,
        )
        .equal(EmptyResult())
    )
