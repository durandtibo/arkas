r"""Contain the Jaccard evaluator for binary labels."""

from __future__ import annotations

__all__ = ["BinaryJaccardEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import BinaryJaccardResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class BinaryJaccardEvaluator(BaseLazyEvaluator[BinaryJaccardResult]):
    r"""Implement the Jaccard evaluator for binary labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_pred: The key or column name of the predicted labels.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> import polars as pl
    >>> from arkas.evaluator import BinaryJaccardEvaluator
    >>> evaluator = BinaryJaccardEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    BinaryJaccardEvaluator(y_true=target, y_pred=pred, drop_nulls=True)
    >>> data = pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 0, 1, 1]})
    >>> result = evaluator.evaluate(data)
    >>> result
    BinaryJaccardResult(y_true=(5,), y_pred=(5,), nan_policy='propagate')

    ```
    """

    def __init__(self, y_true: str, y_pred: str, drop_nulls: bool = True) -> None:
        super().__init__(drop_nulls=drop_nulls)
        self._y_true = y_true
        self._y_pred = y_pred

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred}, "
            f"drop_nulls={self._drop_nulls})"
        )

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> BinaryJaccardResult | Result:
        logger.info(
            f"Evaluating the binary Jaccard | y_true={self._y_true} | y_pred={self._y_pred} | "
            f"drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> BinaryJaccardResult:
        return BinaryJaccardResult(
            y_true=to_array(data[self._y_true]).ravel(), y_pred=to_array(data[self._y_pred]).ravel()
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_pred)
