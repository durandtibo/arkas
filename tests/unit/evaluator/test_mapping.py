from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

from arkas.evaluator import (
    BinaryPrecisionEvaluator,
    BinaryRecallEvaluator,
    MappingEvaluator,
)
from arkas.result import (
    BinaryPrecisionResult,
    BinaryRecallResult,
    MappingResult,
    Result,
)

######################################
#     Tests for MappingEvaluator     #
######################################


def test_mapping_evaluator_repr() -> None:
    assert repr(
        MappingEvaluator(
            {
                "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("MappingEvaluator(")


def test_mapping_evaluator_str() -> None:
    assert str(
        MappingEvaluator(
            {
                "precision": BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                "recall": BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("MappingEvaluator(")


def test_mapping_evaluator_evaluate() -> None:
    assert (
        MappingEvaluator(
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
    result = MappingEvaluator(
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
