from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import MulticlassAveragePrecisionEvaluator
from arkas.result import EmptyResult, MulticlassAveragePrecisionResult, Result

#########################################################
#     Tests for MulticlassAveragePrecisionEvaluator     #
#########################################################


def test_multiclass_average_precision_evaluator_repr() -> None:
    assert repr(MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "MulticlassAveragePrecisionEvaluator("
    )


def test_multiclass_average_precision_evaluator_str() -> None:
    assert str(MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred")).startswith(
        "MulticlassAveragePrecisionEvaluator("
    )


def test_multiclass_average_precision_evaluator_evaluate() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred")
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
            MulticlassAveragePrecisionResult(
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


def test_multiclass_average_precision_evaluator_evaluate_lazy_false() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred")
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
            ),
            lazy=False,
        )
        .equal(
            Result(
                metrics={
                    "average_precision": np.array([1.0, 1.0, 1.0]),
                    "count": 6,
                    "macro_average_precision": 1.0,
                    "micro_average_precision": 1.0,
                    "weighted_average_precision": 1.0,
                }
            )
        )
    )


def test_multiclass_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="prediction")
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
        .equal(EmptyResult())
    )


def test_multiclass_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="missing")
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
            ),
            lazy=False,
        )
        .equal(EmptyResult())
    )


def test_multiclass_average_precision_evaluator_evaluate_multi_arrays() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score=["pred1", "pred2", "pred3"])
        .evaluate(
            {
                "pred1": np.array([0.7, 0.4, 0.1, 0.2, 0.3, 0.1]),
                "pred2": np.array([0.2, 0.3, 0.8, 0.5, 0.2, 0.2]),
                "pred3": np.array([0.1, 0.3, 0.1, 0.3, 0.5, 0.7]),
                "target": np.array([0, 0, 1, 1, 2, 2]),
            }
        )
        .equal(
            MulticlassAveragePrecisionResult(
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


def test_multiclass_average_precision_evaluator_evaluate_dataframe_multi_cols() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score=["pred1", "pred2", "pred3"])
        .evaluate(
            pl.DataFrame(
                {
                    "pred1": [0.7, 0.4, 0.1, 0.2, 0.3, 0.1],
                    "pred2": [0.2, 0.3, 0.8, 0.5, 0.2, 0.2],
                    "pred3": [0.1, 0.3, 0.1, 0.3, 0.5, 0.7],
                    "target": [0, 0, 1, 1, 2, 2],
                }
            )
        )
        .equal(
            MulticlassAveragePrecisionResult(
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
