from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassRocAucEvaluator
from arkas.result import EmptyResult, MulticlassRocAucResult, Result

###############################################
#     Tests for MulticlassRocAucEvaluator     #
###############################################


def test_multiclass_roc_auc_evaluator_repr() -> None:
    assert repr(MulticlassRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "MulticlassRocAucEvaluator("
    )


def test_multiclass_roc_auc_evaluator_str() -> None:
    assert str(MulticlassRocAucEvaluator(y_true="target", y_score="pred")).startswith(
        "MulticlassRocAucEvaluator("
    )


def test_multiclass_roc_auc_evaluator_evaluate() -> None:
    assert (
        MulticlassRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {
                "pred": np.array(
                    [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                "target": np.array([0, 0, 1, 1, 2, 2]),
            }
        )
        .equal(
            MulticlassRocAucResult(
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
            )
        )
    )


def test_multiclass_roc_auc_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            {
                "pred": np.array(
                    [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                "target": np.array([0, 0, 1, 1, 2, 2]),
            },
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "count": 6,
                    "macro_roc_auc": 1.0,
                    "micro_roc_auc": 1.0,
                    "roc_auc": np.array([1.0, 1.0, 1.0]),
                    "weighted_roc_auc": 1.0,
                }
            )
        )
    )


def test_multiclass_roc_auc_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassRocAucEvaluator(y_true="target", y_score="prediction")
        .evaluate(
            {
                "pred": np.array(
                    [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                "target": np.array([0, 0, 1, 1, 2, 2]),
            }
        )
        .equal(EmptyResult())
    )


def test_multiclass_roc_auc_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassRocAucEvaluator(y_true="target", y_score="missing")
        .evaluate(
            {
                "pred": np.array(
                    [
                        [0.7, 0.2, 0.1],
                        [0.4, 0.3, 0.3],
                        [0.1, 0.8, 0.1],
                        [0.2, 0.5, 0.3],
                        [0.3, 0.2, 0.5],
                        [0.1, 0.2, 0.7],
                    ]
                ),
                "target": np.array([0, 0, 1, 1, 2, 2]),
            },
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multiclass_roc_auc_evaluator_evaluate_dataframe() -> None:
    assert (
        MulticlassRocAucEvaluator(y_true="target", y_score="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": np.array(
                        [
                            [0.7, 0.2, 0.1],
                            [0.4, 0.3, 0.3],
                            [0.1, 0.8, 0.1],
                            [0.2, 0.5, 0.3],
                            [0.3, 0.2, 0.5],
                            [0.1, 0.2, 0.7],
                        ]
                    ),
                    "target": [0, 0, 1, 1, 2, 2],
                }
            )
        )
        .equal(
            MulticlassRocAucResult(
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
            )
        )
    )
