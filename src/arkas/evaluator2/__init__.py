r"""Contain data evaluators."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator", "BaseEvaluator", "Evaluator"]

from arkas.evaluator2.accuracy import AccuracyEvaluator
from arkas.evaluator2.base import BaseEvaluator
from arkas.evaluator2.vanilla import Evaluator
