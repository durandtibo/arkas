r"""Implement the balanced accuracy output."""

from __future__ import annotations

__all__ = ["BalancedAccuracyOutput"]


from arkas.content.balanced_accuracy import BalancedAccuracyContentGenerator
from arkas.evaluator2.balanced_accuracy import BalancedAccuracyEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.accuracy import AccuracyState


class BalancedAccuracyOutput(BaseStateOutput[AccuracyState]):
    r"""Implement the balanced accuracy output.

    Args:
        state: The state containing the ground truth and predicted
            labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import BalancedAccuracyOutput
    >>> from arkas.state import AccuracyState
    >>> output = BalancedAccuracyOutput(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> output
    BalancedAccuracyOutput(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> output.get_content_generator()
    BalancedAccuracyContentGenerator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> output.get_evaluator()
    BalancedAccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )

    ```
    """

    def __init__(self, state: AccuracyState) -> None:
        super().__init__(state)
        self._content = BalancedAccuracyContentGenerator(self._state)
        self._evaluator = BalancedAccuracyEvaluator(self._state)

    def _get_content_generator(self) -> BalancedAccuracyContentGenerator:
        return self._content

    def _get_evaluator(self) -> BalancedAccuracyEvaluator:
        return self._evaluator
