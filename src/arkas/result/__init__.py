r"""Contain results."""

from __future__ import annotations

__all__ = [
    "AccuracyResult",
    "AveragePrecisionResult",
    "BalancedAccuracyResult",
    "BaseResult",
    "BinaryAveragePrecisionResult",
    "BinaryClassificationResult",
    "BinaryConfusionMatrixResult",
    "BinaryFbetaScoreResult",
    "BinaryJaccardResult",
    "BinaryPrecisionResult",
    "BinaryRecallResult",
    "BinaryRocAucResult",
    "EmptyResult",
    "EnergyDistanceResult",
    "JensenShannonDivergenceResult",
    "MappingResult",
    "MeanAbsoluteErrorResult",
    "MeanSquaredErrorResult",
    "MulticlassAveragePrecisionResult",
    "MulticlassConfusionMatrixResult",
    "MulticlassFbetaScoreResult",
    "MulticlassJaccardResult",
    "MulticlassPrecisionResult",
    "MulticlassRecallResult",
    "MulticlassRocAucResult",
    "MultilabelAveragePrecisionResult",
    "MultilabelConfusionMatrixResult",
    "MultilabelFbetaScoreResult",
    "MultilabelJaccardResult",
    "MultilabelPrecisionResult",
    "MultilabelRecallResult",
    "MultilabelRocAucResult",
    "PearsonCorrelationResult",
    "PrecisionResult",
    "RecallResult",
    "Result",
    "SequentialResult",
    "SpearmanCorrelationResult",
    "WassersteinDistanceResult",
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
from arkas.result.energy import EnergyDistanceResult
from arkas.result.fbeta import (
    BinaryFbetaScoreResult,
    MulticlassFbetaScoreResult,
    MultilabelFbetaScoreResult,
)
from arkas.result.jaccard import (
    BinaryJaccardResult,
    MulticlassJaccardResult,
    MultilabelJaccardResult,
)
from arkas.result.js import JensenShannonDivergenceResult
from arkas.result.mae import MeanAbsoluteErrorResult
from arkas.result.mapping import MappingResult
from arkas.result.mse import MeanSquaredErrorResult
from arkas.result.pearson import PearsonCorrelationResult
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
from arkas.result.roc_auc import (
    BinaryRocAucResult,
    MulticlassRocAucResult,
    MultilabelRocAucResult,
)
from arkas.result.sequential import SequentialResult
from arkas.result.spearman import SpearmanCorrelationResult
from arkas.result.vanilla import EmptyResult, Result
from arkas.result.wasserstein import WassersteinDistanceResult
