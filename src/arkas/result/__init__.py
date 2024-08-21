r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BaseResult",
    "BinaryClassificationResult",
    "Result",
    "EmptyResult",
    "BalancedAccuracyResult",
    "MergedResult",
]

from arkas.result.accuracy import AccuracyResult, BalancedAccuracyResult
from arkas.result.ap import AveragePrecisionResult
from arkas.result.base import BaseResult
from arkas.result.binary_classification import BinaryClassificationResult
from arkas.result.merge import MergedResult
from arkas.result.vanilla import EmptyResult, Result
