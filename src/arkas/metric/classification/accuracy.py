r"""Implement the classification accuracy metrics."""

from __future__ import annotations

__all__ = ["AccuracyMetric", "accuracy", "balanced_accuracy"]


from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping
from sklearn import metrics

from arkas.metric.base import BaseMetric
from arkas.metric.utils import check_nan_policy, contains_nan, preprocess_pred

if TYPE_CHECKING:
    import numpy as np

    from arkas.state.accuracy import AccuracyState


class AccuracyMetric(BaseMetric):
    r"""Implement the accuracy metric.

    Args:
        state: The state containing the ground truth and predicted
            labels.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import AccuracyMetric
    >>> from arkas.state import AccuracyState
    >>> metric = AccuracyMetric(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> metric
    AccuracyMetric(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )
    >>> metric.evaluate()
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

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
        return accuracy(
            y_true=self._state.y_true,
            y_pred=self._state.y_pred,
            prefix=prefix,
            suffix=suffix,
            nan_policy=self._nan_policy,
        )


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    nan_policy: str = "propagate",
) -> dict[str, float]:
    r"""Return the accuracy metrics.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import accuracy
    >>> accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=nan_policy == "omit"
    )
    y_true_nan = contains_nan(arr=y_true, nan_policy=nan_policy, name="'y_true'")
    y_pred_nan = contains_nan(arr=y_pred, nan_policy=nan_policy, name="'y_pred'")

    count = y_true.size
    acc, correct = float("nan"), float("nan")
    if count > 0 and not y_true_nan and not y_pred_nan:
        correct = int(metrics.accuracy_score(y_true=y_true, y_pred=y_pred, normalize=False))
        acc = float(correct / count)
    return {
        f"{prefix}accuracy{suffix}": acc,
        f"{prefix}count_correct{suffix}": correct,
        f"{prefix}count_incorrect{suffix}": count - correct,
        f"{prefix}count{suffix}": count,
        f"{prefix}error{suffix}": 1.0 - acc,
    }


def balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
    nan_policy: str = "propagate",
) -> dict[str, float]:
    r"""Return the accuracy metrics.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import balanced_accuracy
    >>> balanced_accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'balanced_accuracy': 1.0, 'count': 5}

    ```
    """
    y_true, y_pred = preprocess_pred(
        y_true=y_true.ravel(), y_pred=y_pred.ravel(), drop_nan=nan_policy == "omit"
    )
    y_true_nan = contains_nan(arr=y_true, nan_policy=nan_policy, name="'y_true'")
    y_pred_nan = contains_nan(arr=y_pred, nan_policy=nan_policy, name="'y_pred'")

    count = y_true.size
    acc = float("nan")
    if count > 0 and not y_true_nan and not y_pred_nan:
        acc = float(metrics.balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
    return {f"{prefix}balanced_accuracy{suffix}": acc, f"{prefix}count{suffix}": count}
