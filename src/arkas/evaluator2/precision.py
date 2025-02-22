r"""Implement the precision evaluator."""

from __future__ import annotations

__all__ = ["PrecisionEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.metric import precision
from arkas.state.precision_recall import PrecisionRecallState


class PrecisionEvaluator(BaseStateCachedEvaluator[PrecisionRecallState]):
    r"""Implement the precision evaluator.

    This evaluator can be used in 3 different settings:

    - binary: ``y_true`` must be an array of shape ``(n_samples,)``
        with ``0`` and ``1`` values, and ``y_pred`` must be an array
        of shape ``(n_samples,)``.
    - multiclass: ``y_true`` must be an array of shape ``(n_samples,)``
        with values in ``{0, ..., n_classes-1}``, and ``y_pred`` must
        be an array of shape ``(n_samples,)``.
    - multilabel: ``y_true`` must be an array of shape
        ``(n_samples, n_classes)`` with ``0`` and ``1`` values, and
        ``y_pred`` must be an array of shape
        ``(n_samples, n_classes)``.

    Args:
        state: The state containing the ground truth and predicted
            labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator2 import PrecisionEvaluator
    >>> from arkas.state import PrecisionRecallState
    >>> # binary
    >>> evaluator = PrecisionEvaluator(
    ...     PrecisionRecallState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...         label_type="binary",
    ...     ),
    ... )
    >>> evaluator
    PrecisionEvaluator(
      (state): PrecisionRecallState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', label_type='binary', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'count': 5, 'precision': 1.0}
    >>> # multilabel
    >>> evaluator = PrecisionEvaluator(
    ...     PrecisionRecallState(
    ...         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...         label_type="multilabel",
    ...     )
    ... )
    >>> evaluator
    PrecisionEvaluator(
      (state): PrecisionRecallState(y_true=(5, 3), y_pred=(5, 3), y_true_name='target', y_pred_name='pred', label_type='multilabel', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'count': 5,
     'macro_precision': 0.666...,
     'micro_precision': 0.714...,
     'precision': array([1., 1., 0.]),
     'weighted_precision': 0.625}
    >>> # multiclass
    >>> evaluator = PrecisionEvaluator(
    ...     PrecisionRecallState(
    ...         y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...         y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...         label_type="multiclass",
    ...     ),
    ... )
    >>> evaluator
    PrecisionEvaluator(
      (state): PrecisionRecallState(y_true=(6,), y_pred=(6,), y_true_name='target', y_pred_name='pred', label_type='multiclass', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'count': 6,
     'macro_precision': 1.0,
     'micro_precision': 1.0,
     'precision': array([1., 1., 1.]),
     'weighted_precision': 1.0}
    >>> # auto
    >>> evaluator = PrecisionEvaluator(
    ...     PrecisionRecallState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> evaluator
    PrecisionEvaluator(
      (state): PrecisionRecallState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', label_type='binary', nan_policy='propagate')
    )
    >>> evaluator.evaluate()
    {'count': 5, 'precision': 1.0}

    ```
    """

    def _evaluate(self) -> dict[str, float]:
        return precision(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            label_type=self._state.label_type,
            nan_policy=self._state.nan_policy,
        )
