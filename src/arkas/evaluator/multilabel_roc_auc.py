r"""Contain the Area Under the Receiver Operating Characteristic Curve
(ROC AUC) evaluator for multilabel labels."""

from __future__ import annotations

__all__ = ["MultilabelRocAucEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import MultilabelRocAucResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    import polars as pl


logger = logging.getLogger(__name__)


class MultilabelRocAucEvaluator(BaseLazyEvaluator[MultilabelRocAucResult]):
    r"""Implement the Area Under the Receiver Operating Characteristic
    Curve (ROC AUC) evaluator for multilabel labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_score`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import MultilabelRocAucEvaluator
    >>> evaluator = MultilabelRocAucEvaluator(y_true="target", y_score="pred")
    >>> evaluator
    MultilabelRocAucEvaluator(y_true=target, y_score=pred, drop_nulls=True)
    >>> data = pl.DataFrame(
    ...     {
    ...         "pred": [[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]],
    ...         "target": [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]],
    ...     },
    ...     schema={"pred": pl.Array(pl.Int64, 3), "target": pl.Array(pl.Int64, 3)},
    ... )
    >>> result = evaluator.evaluate(data)
    >>> result
    MultilabelRocAucResult(y_true=(5, 3), y_score=(5, 3))

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

    def evaluate(self, data: pl.DataFrame, lazy: bool = True) -> MultilabelRocAucResult | Result:
        logger.info(
            f"Evaluating the multilabel ROC AUC | y_true={self._y_true} | "
            f"y_score={self._y_score} | drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> MultilabelRocAucResult:
        return MultilabelRocAucResult(
            y_true=to_array(data[self._y_true]), y_score=to_array(data[self._y_score])
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_score)
