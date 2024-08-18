r"""Contain an accuracy evaluator."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.base import BaseLazyEvaluator
from arkas.result import AccuracyResult

if TYPE_CHECKING:
    from arkas.result import BaseResult

logger = logging.getLogger(__name__)


class AccuracyEvaluator(BaseLazyEvaluator):
    r"""Implement the accuracy evaluator.

    Args:
        y_true: The key of the ground truth target labels.
        y_pred: The key of the predicted labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator import AccuracyEvaluator
    >>> data = {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}
    >>> evaluator = AccuracyEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    AccuracyEvaluator(y_true=target, y_pred=pred)
    >>> result = evaluator.evaluate(data)
    >>> result
    AccuracyResult(y_true=(5,), y_pred=(5,))

    ```
    """

    def __init__(self, y_true: str, y_pred: str) -> None:
        self._y_true = y_true
        self._y_pred = y_pred

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred})"

    def _evaluate(self, data: dict) -> BaseResult:
        return AccuracyResult(y_true=data[self._y_true], y_pred=data[self._y_pred])
