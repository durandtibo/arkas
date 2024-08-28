r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BalancedAccuracyResult",
    "BaseResult",
    "BinaryClassificationResult",
    "BinaryPrecisionResult",
    "EmptyResult",
    "MergedResult",
    "MulticlassPrecisionResult",
    "MultilabelPrecisionResult",
    "PrecisionResult",
    "RecallResult",
    "Result",
]

from arkas.result.accuracy import AccuracyResult, BalancedAccuracyResult
from arkas.result.ap import AveragePrecisionResult
from arkas.result.base import BaseResult
from arkas.result.binary_classification import BinaryClassificationResult
from arkas.result.merge import MergedResult
from arkas.result.precision import (
    BinaryPrecisionResult,
    MulticlassPrecisionResult,
    MultilabelPrecisionResult,
    PrecisionResult,
)
from arkas.result.recall import RecallResult
from arkas.result.vanilla import EmptyResult, Result
