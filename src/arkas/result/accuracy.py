r"""Implement the accuracy result."""

from __future__ import annotations

__all__ = ["AccuracyResult", "BalancedAccuracyResult", "accuracy_metrics"]

from typing import Any

import numpy as np
from coola import objects_are_equal
from sklearn import metrics

from arkas.result.base import BaseResult


class AccuracyResult(BaseResult):
    r"""Implement the accuracy result.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` where the values
            are in ``{0, ..., n_classes-1}``.
        y_pred: The predicted labels. This input must be an
            array of shape ``(n_samples,)`` where the values are
            in ``{0, ..., n_classes-1}``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AccuracyResult
    >>> result = AccuracyResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    AccuracyResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()

        self._check_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, y_pred={self._y_pred.shape})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return accuracy_metrics(
            y_true=self._y_true, y_pred=self._y_pred, prefix=prefix, suffix=suffix
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.y_true, other.y_true, equal_nan=equal_nan
        ) and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.shape != self._y_pred.shape:
            msg = (
                f"'y_true' and 'y_pred' have different shapes: {self._y_true.shape} vs "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)


class BalancedAccuracyResult(BaseResult):
    r"""Implement the balanced accuracy result.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` where the values
            are in ``{0, ..., n_classes-1}``.
        y_pred: The predicted labels. This input must be an
            array of shape ``(n_samples,)`` where the values are
            in ``{0, ..., n_classes-1}``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AccuracyResult
    >>> result = BalancedAccuracyResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    BalancedAccuracyResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'balanced_accuracy': 1.0, 'count': 5}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()

        self._check_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, y_pred={self._y_pred.shape})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        count = self._y_true.size
        accuracy = float("nan")
        if count > 0:
            accuracy = float(
                metrics.balanced_accuracy_score(y_true=self._y_true, y_pred=self._y_pred)
            )
        return {f"{prefix}balanced_accuracy{suffix}": accuracy, f"{prefix}count{suffix}": count}

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.y_true, other.y_true, equal_nan=equal_nan
        ) and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.shape != self._y_pred.shape:
            msg = (
                f"'y_true' and 'y_pred' have different shapes: {self._y_true.shape} vs "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)


def accuracy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    prefix: str = "",
    suffix: str = "",
) -> dict[str, float]:
    r"""Return the accuracy metrics.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.
        prefix: The key prefix in the returned dictionary.
        suffix: The key suffix in the returned dictionary.

    Returns:
        The computed metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result.accuracy import accuracy_metrics
    >>> accuracy_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    {'accuracy': 1.0, 'count_correct': 5, 'count_incorrect': 0, 'count': 5, 'error': 0.0}

    ```
    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()

    # Ignore samples that have NaN values.
    mask = np.logical_not(np.logical_or(np.isnan(y_true), np.isnan(y_pred)))
    y_true, y_pred = y_true[mask], y_pred[mask]

    count = y_true.size
    count_correct = int(metrics.accuracy_score(y_true=y_true, y_pred=y_pred, normalize=False))
    accuracy = float(count_correct / count) if count > 0 else float("nan")
    return {
        f"{prefix}accuracy{suffix}": accuracy,
        f"{prefix}count_correct{suffix}": count_correct,
        f"{prefix}count_incorrect{suffix}": count - count_correct,
        f"{prefix}count{suffix}": count,
        f"{prefix}error{suffix}": 1.0 - accuracy,
    }
