r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BalancedAccuracyResult",
    "BaseResult",
    "BinaryClassificationResult",
    "BinaryPrecisionResult",
    "BinaryRecallResult",
    "EmptyResult",
    "SequentialResult",
    "MulticlassPrecisionResult",
    "MulticlassRecallResult",
    "MultilabelPrecisionResult",
    "PrecisionResult",
    "RecallResult",
    "Result",
    "MultilabelRecallResult",
    "BinaryAveragePrecisionResult",
    "MulticlassAveragePrecisionResult",
    "MultilabelAveragePrecisionResult",
    "MappingResult",
]

from arkas.result.accuracy import AccuracyResult, BalancedAccuracyResult
from arkas.result.ap import (
    AveragePrecisionResult,
    BinaryAveragePrecisionResult,
    MulticlassAveragePrecisionResult,
    MultilabelAveragePrecisionResult,
)
from arkas.result.base import BaseResult
from arkas.result.binary_classification import BinaryClassificationResult
from arkas.result.mapping import MappingResult
from arkas.result.precision import (
    BinaryPrecisionResult,
    MulticlassPrecisionResult,
    MultilabelPrecisionResult,
    PrecisionResult,
)
from arkas.result.recall import (
    BinaryRecallResult,
    MulticlassRecallResult,
    MultilabelRecallResult,
    RecallResult,
)
from arkas.result.sequential import SequentialResult
from arkas.result.vanilla import EmptyResult, Result
