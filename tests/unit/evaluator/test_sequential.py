from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal
from objectory import OBJECT_TARGET

from arkas.evaluator import (
    BinaryPrecisionEvaluator,
    BinaryRecallEvaluator,
    SequentialEvaluator,
)
from arkas.result import (
    BinaryPrecisionResult,
    BinaryRecallResult,
    Result,
    SequentialResult,
)

#########################################
#     Tests for SequentialEvaluator     #
#########################################


def test_sequential_evaluator_repr() -> None:
    assert repr(
        SequentialEvaluator(
            [
                BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            ]
        )
    ).startswith("SequentialEvaluator(")


def test_sequential_evaluator_str() -> None:
    assert str(
        SequentialEvaluator(
            [
                BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            ]
        )
    ).startswith("SequentialEvaluator(")


def test_sequential_evaluator_evaluate() -> None:
    assert (
        SequentialEvaluator(
            [
                BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                {
                    OBJECT_TARGET: "arkas.evaluator.BinaryRecallEvaluator",
                    "y_true": "target",
                    "y_pred": "pred",
                },
            ]
        )
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            SequentialResult(
                [
                    BinaryPrecisionResult(
                        y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
                    ),
                    BinaryRecallResult(
                        y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
                    ),
                ]
            )
        )
    )


def test_sequential_evaluator_evaluate_lazy_false() -> None:
    result = SequentialEvaluator(
        [
            BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
            BinaryRecallEvaluator(y_true="target", y_pred="pred"),
        ]
    ).evaluate(pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
    assert isinstance(result, Result)
    assert objects_are_equal(
        result.compute_metrics(), {"count": 5, "precision": 1.0, "recall": 1.0}
    )
    assert len(result.generate_figures()) == 1
