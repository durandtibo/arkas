r"""Contain evaluators."""

from __future__ import annotations

__all__ = [
    "AccuracyEvaluator",
    "AveragePrecisionEvaluator",
    "BaseEvaluator",
    "BaseLazyEvaluator",
    "BinaryPrecisionEvaluator",
    "MulticlassPrecisionEvaluator",
    "is_evaluator_config",
    "setup_evaluator",
]

from arkas.evaluator.accuracy import AccuracyEvaluator
from arkas.evaluator.ap import AveragePrecisionEvaluator
from arkas.evaluator.base import (
    BaseEvaluator,
    BaseLazyEvaluator,
    is_evaluator_config,
    setup_evaluator,
)
from arkas.evaluator.binary_precision import BinaryPrecisionEvaluator
from arkas.evaluator.multiclass_precision import MulticlassPrecisionEvaluator
