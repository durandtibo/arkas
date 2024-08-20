r"""Contain an accuracy evaluator."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.base import BaseLazyEvaluator
from arkas.result import AccuracyResult, EmptyResult
from arkas.utils.array import to_array
from arkas.utils.mapping import find_missing_keys

if TYPE_CHECKING:
    import polars as pl

    from arkas.result import BaseResult

logger = logging.getLogger(__name__)


class AccuracyEvaluator(BaseLazyEvaluator):
    r"""Implement the accuracy evaluator.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_pred: The key or column name of the predicted labels.

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
    >>> frame = pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})
    >>> result = evaluator.evaluate(frame)
    >>> result
    AccuracyResult(y_true=(6,), y_pred=(6,))

    ```
    """

    def __init__(self, y_true: str, y_pred: str) -> None:
        self._y_true = y_true
        self._y_pred = y_pred

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred})"

    def _evaluate(self, data: dict | pl.DataFrame) -> BaseResult:
        logger.info(f"Evaluating the accuracy | y_true={self._y_true} | y_pred={self._y_pred}")
        if missing_keys := find_missing_keys(data, keys=[self._y_pred, self._y_true]):
            logger.warning(
                "Skipping the accuracy evaluation because some keys are missing: "
                f"{sorted(missing_keys)}"
            )
            return EmptyResult()
        return AccuracyResult(
            y_true=to_array(data[self._y_true]), y_pred=to_array(data[self._y_pred])
        )


class AccuracyDataFrameEvaluator(BaseLazyEvaluator):
    r"""Implement an accuracy evaluator that uses the data from a
    DataFrame.

    Args:
        y_true: The column name of the ground truth target labels.
        y_pred: The column name of the predicted labels.
        in_key: The key of the DataFrame in the input dictionary.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import AccuracyDataFrameEvaluator
    >>> data = {"frame": pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [3, 2, 0, 1, 0]})}
    >>> evaluator = AccuracyDataFrameEvaluator(y_true="target", y_pred="pred", in_key="frame")
    >>> evaluator
    AccuracyDataFrameEvaluator(y_true=target, y_pred=pred, in_key=frame)
    >>> result = evaluator.evaluate(data)
    >>> result
    AccuracyResult(y_true=(5,), y_pred=(5,))

    ```
    """

    def __init__(self, y_true: str, y_pred: str, in_key: str) -> None:
        self._y_true = y_true
        self._y_pred = y_pred
        self._in_key = in_key

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred}, "
            f"in_key={self._in_key})"
        )

    def _evaluate(self, data: dict) -> BaseResult:
        frame = data[self._in_key]
        return AccuracyResult(
            y_true=frame[self._y_true].to_numpy(), y_pred=frame[self._y_pred].to_numpy()
        )
