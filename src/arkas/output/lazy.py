r"""Contain a base class that partially implements the lazy computation
logic."""

from __future__ import annotations

__all__ = ["BaseLazyOutput"]

from abc import abstractmethod
from typing import TYPE_CHECKING

from arkas.evaluator2.vanilla import Evaluator
from arkas.output.base import BaseOutput

if TYPE_CHECKING:
    from arkas.evaluator2.base import BaseEvaluator


class BaseLazyOutput(BaseOutput):
    r"""Define a base class that partially implements the lazy
    computation logic."""

    def get_evaluator(self, lazy: bool = True) -> BaseEvaluator:
        evaluator = self._get_evaluator()
        if lazy:
            return evaluator
        return Evaluator(metrics=evaluator.evaluate())

    @abstractmethod
    def _get_evaluator(self) -> BaseEvaluator:
        r"""Get the evaluator associated to the output.

        Returns:
            The evaluator.
        """
