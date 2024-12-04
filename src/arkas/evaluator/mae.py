r"""Contain the mean absolute error (MAE) evaluator."""

from __future__ import annotations

__all__ = ["MeanAbsoluteErrorEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import MeanAbsoluteErrorResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class MeanAbsoluteErrorEvaluator(BaseLazyEvaluator[MeanAbsoluteErrorResult]):
    r"""Implement the mean absolute error (MAE) evaluator.

    Args:
        y_true: The key or column name of the ground truth target
            values.
        y_pred: The key or column name of the predicted values.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import MeanAbsoluteErrorEvaluator
    >>> evaluator = MeanAbsoluteErrorEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    MeanAbsoluteErrorEvaluator(y_true=target, y_pred=pred, drop_nulls=True)
    >>> data = pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]})
    >>> result = evaluator.evaluate(data)
    >>> result
    MeanAbsoluteErrorResult(y_true=(5,), y_pred=(5,), nan_policy=propagate)

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

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> MeanAbsoluteErrorResult | Result:
        logger.info(
            f"Evaluating the mean absolute error | y_true={self._y_true} | "
            f"y_pred={self._y_pred} | drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> MeanAbsoluteErrorResult:
        return MeanAbsoluteErrorResult(
            y_true=to_array(data[self._y_true]).ravel(), y_pred=to_array(data[self._y_pred]).ravel()
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_pred)
