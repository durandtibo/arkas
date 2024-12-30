r"""Implement the accuracy state."""

from __future__ import annotations

__all__ = ["AccuracyState"]

import sys
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.metric.utils import check_same_shape_pred
from arkas.state.base import BaseState

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )

if TYPE_CHECKING:
    import numpy as np


class AccuracyState(BaseState):
    r"""Implement the accuracy state.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` where the values
            are in ``{0, ..., n_classes-1}``.
        y_pred: The predicted labels. This input must be an
            array of shape ``(n_samples,)`` where the values are
            in ``{0, ..., n_classes-1}``.
        y_true_name: The name associated to the ground truth target
            labels.
        y_pred_name: The name associated to the predicted labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.state import AccuracyState
    >>> state = AccuracyState(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     y_true_name="target",
    ...     y_pred_name="pred",
    ... )
    >>> state
    AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')

    ```
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_true_name: str,
        y_pred_name: str,
    ) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()
        check_same_shape_pred(y_true=self._y_true, y_pred=self._y_pred)

        self._y_true_name = y_true_name
        self._y_pred_name = y_pred_name

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "y_true": self._y_true.shape,
                "y_pred": self._y_pred.shape,
                "y_true_name": self._y_true_name,
                "y_pred_name": self._y_pred_name,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    @property
    def y_true_name(self) -> str:
        return self._y_true_name

    @property
    def y_pred_name(self) -> str:
        return self._y_pred_name

    def clone(self, deep: bool = True) -> Self:
        return self.__class__(
            y_true=self._y_true.copy() if deep else self._y_true,
            y_pred=self._y_pred.copy() if deep else self._y_pred,
            y_true_name=self._y_true_name,
            y_pred_name=self._y_pred_name,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)
            and self.y_true_name == other.y_true_name
            and self.y_pred_name == other.y_pred_name
        )
