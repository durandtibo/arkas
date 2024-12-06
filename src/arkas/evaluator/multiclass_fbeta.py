r"""Contain the F-beta evaluator for multiclass labels."""

from __future__ import annotations

__all__ = ["MulticlassFbetaScoreEvaluator"]

import logging
from typing import TYPE_CHECKING

from arkas.evaluator.lazy import BaseLazyEvaluator
from arkas.result import MulticlassFbetaScoreResult, Result
from arkas.utils.array import to_array

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


logger = logging.getLogger(__name__)


class MulticlassFbetaScoreEvaluator(BaseLazyEvaluator[MulticlassFbetaScoreResult]):
    r"""Implement the F-beta evaluator for multiclass labels.

    Args:
        y_true: The key or column name of the ground truth target
            labels.
        y_pred: The key or column name of the predicted labels.
        betas: The betas used to compute the F-beta scores.
        drop_nulls: If ``True``, the rows with null values in
            ``y_true`` or ``y_pred`` columns are dropped.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import MulticlassFbetaScoreEvaluator
    >>> evaluator = MulticlassFbetaScoreEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    MulticlassFbetaScoreEvaluator(y_true=target, y_pred=pred, betas=(1,), drop_nulls=True)
    >>> data = pl.DataFrame({"pred": [0, 0, 1, 1, 2, 2], "target": [0, 0, 1, 1, 2, 2]})
    >>> result = evaluator.evaluate(data)
    >>> result
    MulticlassFbetaScoreResult(y_true=(6,), y_pred=(6,), betas=(1,), nan_policy=propagate)

    ```
    """

    def __init__(
        self, y_true: str, y_pred: str, betas: Sequence[float] = (1,), drop_nulls: bool = True
    ) -> None:
        super().__init__(drop_nulls=drop_nulls)
        self._y_true = y_true
        self._y_pred = y_pred
        self._betas = tuple(betas)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true}, y_pred={self._y_pred}, "
            f"betas={self._betas}, drop_nulls={self._drop_nulls})"
        )

    def evaluate(
        self, data: pl.DataFrame, lazy: bool = True
    ) -> MulticlassFbetaScoreResult | Result:
        logger.info(
            f"Evaluating the multiclass F-beta score | y_true={self._y_true} | "
            f"y_pred={self._y_pred} | drop_nulls={self._drop_nulls}"
        )
        return self._evaluate(data, lazy)

    def _compute_result(self, data: pl.DataFrame) -> MulticlassFbetaScoreResult:
        return MulticlassFbetaScoreResult(
            y_true=to_array(data[self._y_true]).ravel(),
            y_pred=to_array(data[self._y_pred]).ravel(),
            betas=self._betas,
        )

    def _get_columns(self) -> tuple[str, ...]:
        return (self._y_true, self._y_pred)
