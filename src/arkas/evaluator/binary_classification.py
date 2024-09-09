r"""Contain the average precision evaluator for binary labels."""

from __future__ import annotations

__all__ = ["BinaryClassificationEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import BinaryClassificationResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class BinaryClassificationEvaluator(BaseLazyEvaluator[BinaryClassificationResult]):
    r"""Implement the average precision evaluator for binary labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_pred: The key or column name of the predicted labels.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import BinaryClassificationEvaluator
    >>> evaluator = BinaryClassificationEvaluator(
    ...     y_true="target", y_pred="pred", y_score="score"
    ... )
    >>> evaluator
    BinaryClassificationEvaluator(y_true=target, y_pred=pred, y_score=score, drop_nulls=True)
    >>> data = pl.DataFrame(
    ...     {
    ...         "pred": [1, 0, 0, 1, 1],
    ...         "score": [2, -1, 0, 3, 1],
    ...         "target": [1, 0, 0, 1, 1],
    ...     }
    ... )
    >>> result = evaluator.evaluate(data)
    >>> result
    BinaryClassificationResult(y_true=(5,), y_pred=(5,), y_score=(5,), betas=(1,))

    ```
    """

    def __init__(
        self, y_true: str, y_pred: str, y_score: str | None = None, drop_nulls: bool = True
    ) -> None:
        super().__init__(drop_nulls=drop_nulls)
        self._y_true = y_true
        self._y_pred = y_pred
        self._y_score = y_score

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred}, "
            f"y_score={self._y_score}, drop_nulls={self._drop_nulls})"
        )

    def evaluate(
        self, data: pl.DataFrame, lazy: bool = True
    ) -> BinaryClassificationResult | Result:
        logger.info(
            f"Evaluating the binary classification metrics | y_true={self._y_true} | "
            f"y_pred={self._y_pred} | y_score={self._y_score} | drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> BinaryClassificationResult:
        return BinaryClassificationResult(
            y_true=to_array(data[self._y_true]).ravel(),
            y_pred=to_array(data[self._y_pred]).ravel(),
            y_score=to_array(data[self._y_score]).ravel() if self._y_score is not None else None,
        )

    def _get_columns(self) -> tuple[str, ...]:
        if self._y_score is None:
            return (self._y_pred, self._y_true)
        return (self._y_pred, self._y_true, self._y_score)
