r"""Contain an accuracy evaluator."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import AccuracyResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class AccuracyEvaluator(BaseLazyEvaluator[AccuracyResult]):
    r"""Implement the accuracy evaluator.

    Args:
        y_true: The column name of the ground truth target
            labels.
        y_pred: The column name of the predicted labels.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import AccuracyEvaluator
    >>> evaluator = AccuracyEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    AccuracyEvaluator(y_true=target, y_pred=pred, drop_nulls=True)
    >>> frame = pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})
    >>> result = evaluator.evaluate(frame)
    >>> result
    AccuracyResult(y_true=(6,), y_pred=(6,))

    ```
    """

    def __init__(self, y_true: str, y_pred: str, drop_nulls: bool = True) -> None:
        super().__init__(drop_nulls=drop_nulls)
        self._y_true = y_true
        self._y_pred = y_pred

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred}, drop_nulls={self._drop_nulls})"

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> AccuracyResult | Result:
        logger.info(f"Evaluating the accuracy | y_true={self._y_true} | y_pred={self._y_pred}")
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> AccuracyResult:
        return AccuracyResult(
            y_true=to_array(data[self._y_true]).ravel(), y_pred=to_array(data[self._y_pred]).ravel()
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_pred)
