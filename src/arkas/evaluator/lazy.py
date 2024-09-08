r"""Contain the base class to implement a lazy evaluator."""

from __future__ import annotations

__all__ = ["BaseLazyEvaluator"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from arkas.evaluator.base import BaseEvaluator
from arkas.result import EmptyResult, Result
from arkas.utils.data import find_missing_keys

if TYPE_CHECKING:
    import polars as pl

    from arkas.result import BaseResult

logger = logging.getLogger(__name__)


class BaseLazyEvaluator(BaseEvaluator):
    r"""Define the base class to evaluate the result lazily.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import AccuracyEvaluator
    >>> evaluator = AccuracyEvaluator(y_true="target", y_pred="pred")
    >>> evaluator
    AccuracyEvaluator(y_true=target, y_pred=pred)
    >>> frame = pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})
    >>> result = evaluator.evaluate(frame)
    >>> result
    AccuracyResult(y_true=(6,), y_pred=(6,))

    ```
    """

    def _evaluate(self, data: pl.DataFrame, lazy: bool = True) -> BaseResult:
        r"""Evaluate the result.

        Args:
            data: The data to evaluate.
            lazy: If ``True``, it forces the computation of the
                result, otherwise it returns a result object that
                delays the evaluation of the result.

        Returns:
            The generated result.
        """
        if missing_keys := find_missing_keys(keys=set(data.columns), queries=self._get_columns()):
            logger.warning(
                "Skipping the accuracy evaluation because some keys are missing: "
                f"{sorted(missing_keys)}"
            )
            return EmptyResult()

        out = self._compute_result(data)
        if lazy or isinstance(out, EmptyResult):
            return out
        return Result(metrics=out.compute_metrics(), figures=out.generate_figures())

    @abstractmethod
    def _compute_result(self, data: pl.DataFrame) -> BaseResult:
        r"""Compute and return the result.

        Args:
            data: The data to evaluate.

        Returns:
            The generated result.
        """

    @abstractmethod
    def _get_columns(self) -> tuple[str, ...]:
        r"""Get the columns used to compute the result.

        Returns:
            The column names.
        """
