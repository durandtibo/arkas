r"""Implement the balanced accuracy evaluator."""

from __future__ import annotations

__all__ = ["BalancedAccuracyEvaluator"]


from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping

from arkas.evaluator2.base import BaseEvaluator
from arkas.evaluator2.vanilla import Evaluator
from arkas.metric import balanced_accuracy
from arkas.metric.utils import check_nan_policy

if TYPE_CHECKING:
    from arkas.state.accuracy import AccuracyState


class BalancedAccuracyEvaluator(BaseEvaluator):
    r"""Implement the balanced accuracy evaluator.

    Args:
        state: The state containing the ground truth and predicted
            labels.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator2 import BalancedAccuracyEvaluator
    >>> from arkas.state import AccuracyState
    >>> evaluator = BalancedAccuracyEvaluator(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> evaluator
    BalancedAccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )
    >>> evaluator.evaluate()
    {'balanced_accuracy': 1.0, 'count': 5}

    ```
    """

    def __init__(self, state: AccuracyState, nan_policy: str = "propagate") -> None:
        self._state = state
        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state, "nan_policy": self._nan_policy}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self._state.equal(other._state, equal_nan=equal_nan)
            and self._nan_policy == other._nan_policy
        )

    def evaluate(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return balanced_accuracy(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            prefix=prefix,
            suffix=suffix,
            nan_policy=self._nan_policy,
        )

    def precompute(self) -> Evaluator:
        return Evaluator(metrics=self.evaluate())
