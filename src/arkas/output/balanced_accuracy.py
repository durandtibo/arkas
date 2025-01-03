r"""Implement the balanced accuracy output."""

from __future__ import annotations

__all__ = ["BalancedAccuracyOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping

from arkas.content.balanced_accuracy import BalancedAccuracyContentGenerator
from arkas.evaluator2.balanced_accuracy import BalancedAccuracyEvaluator
from arkas.metric.utils import check_nan_policy
from arkas.output.lazy import BaseLazyOutput
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    from arkas.state.accuracy import AccuracyState


class BalancedAccuracyOutput(BaseLazyOutput):
    r"""Implement the balanced accuracy output.

    Args:
        state: The state containing the ground truth and predicted
            labels.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

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
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )
    >>> output.get_content_generator()
    BalancedAccuracyContentGenerator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )
    >>> output.get_evaluator()
    BalancedAccuracyEvaluator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )
    >>> output.get_plotter()
    Plotter(count=0)

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

    def _get_content_generator(self) -> BalancedAccuracyContentGenerator:
        return BalancedAccuracyContentGenerator(state=self._state, nan_policy=self._nan_policy)

    def _get_evaluator(self) -> BalancedAccuracyEvaluator:
        return BalancedAccuracyEvaluator(state=self._state, nan_policy=self._nan_policy)

    def _get_plotter(self) -> Plotter:
        return Plotter()
