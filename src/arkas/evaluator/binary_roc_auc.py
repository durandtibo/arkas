r"""Contain the Area Under the Receiver Operating Characteristic Curve
(ROC AUC) evaluator for binary labels."""

from __future__ import annotations

__all__ = ["BinaryRocAucEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import BinaryRocAucResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class BinaryRocAucEvaluator(BaseLazyEvaluator[BinaryRocAucResult]):
    r"""Implement the Area Under the Receiver Operating Characteristic
    Curve (ROC AUC) evaluator for binary labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import BinaryRocAucEvaluator
    >>> evaluator = BinaryRocAucEvaluator(y_true="target", y_score="pred")
    >>> evaluator
    BinaryRocAucEvaluator(y_true=target, y_score=pred, drop_nulls=True)
    >>> data = pl.DataFrame({"pred": [2, -1, 0, 3, 1], "target": [1, 0, 0, 1, 1]})
    >>> result = evaluator.evaluate(data)
    >>> result
    BinaryRocAucResult(y_true=(5,), y_score=(5,))

    ```
    """

    def __init__(self, y_true: str, y_score: str, drop_nulls: bool = True) -> None:
        super().__init__(drop_nulls=drop_nulls)
        self._y_true = y_true
        self._y_score = y_score

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true}, y_score={self._y_score}, "
            f"drop_nulls={self._drop_nulls})"
        )

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> BinaryRocAucResult | Result:
        logger.info(
            f"Evaluating the binary ROC AUC | y_true={self._y_true} | y_score={self._y_score}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> BinaryRocAucResult:
        return BinaryRocAucResult(
            y_true=to_array(data[self._y_true]).ravel(),
            y_score=to_array(data[self._y_score]).ravel(),
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_score)
