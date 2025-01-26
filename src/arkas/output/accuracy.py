r"""Implement the accuracy output."""

from __future__ import annotations

__all__ = ["AccuracyOutput"]


from arkas.content.accuracy import AccuracyContentGenerator
from arkas.evaluator2.accuracy import AccuracyEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.accuracy import AccuracyState


class AccuracyOutput(BaseStateOutput[AccuracyState]):
    r"""Implement the accuracy output.

    Args:
        state: The state containing the ground truth and predicted
            labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import AccuracyOutput
    >>> from arkas.state import AccuracyState
    >>> output = AccuracyOutput(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> output
    AccuracyOutput(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )
    >>> output.get_content_generator()
    AccuracyContentGenerator(
      (evaluator): AccuracyEvaluator(
          (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
        )
    )
    >>> output.get_evaluator()
    AccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
    )

    ```
    """

    def __init__(self, state: AccuracyState) -> None:
        super().__init__(state)
        self._evaluator = AccuracyEvaluator(self._state)
        self._content = AccuracyContentGenerator(self._evaluator)

    def _get_content_generator(self) -> AccuracyContentGenerator:
        return self._content

    def _get_evaluator(self) -> AccuracyEvaluator:
        return self._evaluator
