r"""Contain results."""

from __future__ import annotations

__all__ = ["AccuracyResult", "BaseResult", "BinaryClassificationResult", "Result"]

from arkas.result.accuracy import AccuracyResult
from arkas.result.base import BaseResult
from arkas.result.binary_classification import BinaryClassificationResult
from arkas.result.vanilla import Result
