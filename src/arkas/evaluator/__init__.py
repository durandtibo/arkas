r"""Contain evaluators."""

from __future__ import annotations

__all__ = [
    "AccuracyEvaluator",
    "AccuracyDataFrameEvaluator",
    "BaseEvaluator",
    "BaseLazyEvaluator",
    "is_evaluator_config",
    "setup_evaluator",
]

from arkas.evaluator.accuracy import AccuracyDataFrameEvaluator, AccuracyEvaluator
from arkas.evaluator.base import (
    BaseEvaluator,
    BaseLazyEvaluator,
    is_evaluator_config,
    setup_evaluator,
)
