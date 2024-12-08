r"""Contain the precision evaluator for binary labels."""

from __future__ import annotations

__all__ = ["BinaryPrecisionEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import BinaryPrecisionResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class BinaryPrecisionEvaluator(BaseLazyEvaluator[BinaryPrecisionResult]):
    r"""Implement the precision evaluator for binary labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_pred: The key or column name of the predicted labels.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import BinaryPrecisionEvaluator
    >>> evaluator = BinaryPrecisionEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    BinaryPrecisionEvaluator(y_true=target, y_pred=pred, drop_nulls=True)
    >>> data = pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 0, 1, 1]})
    >>> result = evaluator.evaluate(data)
    >>> result
    BinaryPrecisionResult(y_true=(5,), y_pred=(5,), nan_policy='propagate')

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

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> BinaryPrecisionResult | Result:
        logger.info(
            f"Evaluating the binary precision | y_true={self._y_true} | y_pred={self._y_pred} | "
            f"drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> BinaryPrecisionResult:
        return BinaryPrecisionResult(
            y_true=to_array(data[self._y_true]).ravel(), y_pred=to_array(data[self._y_pred]).ravel()
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_pred)
