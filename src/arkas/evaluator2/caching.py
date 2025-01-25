r"""Define the base class to implement an evaluator that cache the
metrics after the first evaluation."""

from __future__ import annotations

__all__ = ["BaseCacheEvaluator"]

from abc import abstractmethod

from arkas.evaluator2.base import BaseEvaluator


class BaseCacheEvaluator(BaseEvaluator):
    r"""Define the base class to implement an evaluator that cache the
    metrics after the first evaluation."""

    def __init__(self) -> None:
        self._cached_metrics = None

    def evaluate(self, prefix: str = "", suffix: str = "") -> dict:
        if self._cached_metrics is None:
            self._cached_metrics = self._evaluate()
        return {f"{prefix}{col}{suffix}": val for col, val in self._cached_metrics.items()}

    @abstractmethod
    def _evaluate(self) -> dict:
        r"""Evaluate the metrics.

        Returns:
            The metrics.
        """
