r"""Define the base class to implement an evaluator that caches the
metrics after the first evaluation."""

from __future__ import annotations

__all__ = ["BaseCacheEvaluator", "BaseStateCachedEvaluator"]

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.evaluator2.base import BaseEvaluator
from arkas.evaluator2.vanilla import Evaluator
from arkas.state.base import BaseState

T = TypeVar("T", bound=BaseState)


class BaseCacheEvaluator(BaseEvaluator):
    r"""Define the base class to implement an evaluator that caches the
    metrics after the first evaluation."""

    def __init__(self) -> None:
        self._cached_metrics = None

    def compute(self) -> Evaluator:
        return Evaluator(metrics=self.evaluate())

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


class BaseStateCachedEvaluator(BaseCacheEvaluator, Generic[T]):
    r"""Define the base class to implement an evaluator that caches the
    metrics after the first evaluation, and computes the metrics from a
    state object.

    Args:
        state: The state with the data.
    """

    def __init__(self, state: T) -> None:
        super().__init__()
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def state(self) -> T:
        return self._state

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)
