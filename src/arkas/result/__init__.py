r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BaseResult",
    "BinaryClassificationResult",
    "Result",
    "EmptyResult",
]

from arkas.result.accuracy import AccuracyResult
from arkas.result.ap import AveragePrecisionResult
from arkas.result.base import BaseResult
from arkas.result.binary_classification import BinaryClassificationResult
from arkas.result.vanilla import EmptyResult, Result
