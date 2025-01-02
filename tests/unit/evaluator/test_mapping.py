from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

from arkas.evaluator import (
    BinaryPrecisionEvaluator,
    BinaryRecallEvaluator,
    EvaluatorDict,
)
from arkas.result import (
    BinaryPrecisionResult,
    BinaryRecallResult,
    MappingResult,
    Result,
)

######################################
#     Tests for EvaluatorDict     #
######################################


def test_mapping_evaluator_repr() -> None:
    assert repr(
        EvaluatorDict(
            {
                "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("EvaluatorDict(")


def test_mapping_evaluator_str() -> None:
    assert str(
        EvaluatorDict(
            {
                "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("EvaluatorDict(")


def test_mapping_evaluator_evaluate() -> None:
    assert (
        EvaluatorDict(
            {
                "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            }
        )
        .evaluate(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            MappingResult(
                {
                    "precision": BinaryPrecisionResult(
                        y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
                    ),
                    "recall": BinaryRecallResult(
                        y_true=np.array([1, 0, 1, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1])
                    ),
                }
            )
        )
    )


def test_mapping_evaluator_evaluate_lazy_false() -> None:
    result = EvaluatorDict(
        {
            "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
            "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
        }
    ).evaluate(pl.DataFrame({"pred": [1, 0, 1, 0, 1], "target": [1, 0, 1, 0, 1]}), lazy=False)
    assert isinstance(result, Result)
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": {"count": 5, "precision": 1.0},
            "recall": {"count": 5, "recall": 1.0},
        },
    )
    assert len(result.generate_figures()) == 2
