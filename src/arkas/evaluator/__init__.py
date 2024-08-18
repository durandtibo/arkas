r"""Contain evaluators."""

from __future__ import annotations

__all__ = ["BaseEvaluator", "setup_evaluator", "is_evaluator_config"]

from arkas.evaluator.base import BaseEvaluator, is_evaluator_config, setup_evaluator
