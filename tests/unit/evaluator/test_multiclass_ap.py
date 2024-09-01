from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

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
    result = MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred").evaluate(
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
    assert isinstance(result, Result)
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_average_precision_evaluator_evaluate_missing_keys() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="prediction")
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


def test_multiclass_average_precision_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="missing")
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


def test_multiclass_average_precision_evaluator_evaluate_dataframe() -> None:
    assert (
        MulticlassAveragePrecisionEvaluator(y_true="target", y_score="pred")
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
