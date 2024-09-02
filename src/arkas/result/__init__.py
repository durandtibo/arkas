r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BalancedAccuracyResult",
    "BaseResult",
    "BinaryAveragePrecisionResult",
    "BinaryClassificationResult",
    "BinaryJaccardResult",
    "BinaryPrecisionResult",
    "BinaryRecallResult",
    "EmptyResult",
    "MappingResult",
    "MulticlassAveragePrecisionResult",
    "MulticlassJaccardResult",
    "MulticlassPrecisionResult",
    "MulticlassRecallResult",
    "MultilabelAveragePrecisionResult",
    "MultilabelJaccardResult",
    "MultilabelPrecisionResult",
    "MultilabelRecallResult",
    "PrecisionResult",
    "RecallResult",
    "Result",
    "SequentialResult",
    "BinaryConfusionMatrixResult",
    "MulticlassConfusionMatrixResult",
    "MultilabelConfusionMatrixResult",
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
from arkas.result.confmat import (
    BinaryConfusionMatrixResult,
    MulticlassConfusionMatrixResult,
    MultilabelConfusionMatrixResult,
)
from arkas.result.jaccard import (
    BinaryJaccardResult,
    MulticlassJaccardResult,
    MultilabelJaccardResult,
)
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
