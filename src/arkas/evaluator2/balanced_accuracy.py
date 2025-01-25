r"""Implement the balanced accuracy evaluator."""

from __future__ import annotations

__all__ = ["BalancedAccuracyEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.metric import balanced_accuracy
from arkas.state.accuracy import AccuracyState


class BalancedAccuracyEvaluator(BaseStateCachedEvaluator[AccuracyState]):
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
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'balanced_accuracy': 1.0, 'count': 5}

    ```
    """

    def _evaluate(self) -> dict[str, float]:
        return balanced_accuracy(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            nan_policy=self._state.nan_policy,
        )
