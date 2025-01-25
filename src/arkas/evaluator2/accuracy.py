r"""Implement the accuracy evaluator."""

from __future__ import annotations

__all__ = ["AccuracyEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.metric import accuracy
from arkas.state.accuracy import AccuracyState


class AccuracyEvaluator(BaseStateCachedEvaluator[AccuracyState]):
    r"""Implement the accuracy evaluator.

    Args:
        state: The state containing the ground truth and predicted
            labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator2 import AccuracyEvaluator
    >>> from arkas.state import AccuracyState
    >>> evaluator = AccuracyEvaluator(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> evaluator
    AccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """

    def _evaluate(self) -> dict[str, float]:
        return accuracy(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            nan_policy=self._state.nan_policy,
        )
