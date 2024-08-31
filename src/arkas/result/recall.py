r"""Implement the recall result."""

from __future__ import annotations

__all__ = [
    "BaseRecallResult",
    "BinaryRecallResult",
    "MulticlassRecallResult",
    "RecallResult",
    "MultilabelRecallResult",
]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal

from arkas.metric import recall_metrics
from arkas.metric.figure import binary_precision_recall_curve
from arkas.metric.recall import (
    binary_recall_metrics,
    find_label_type,
    multiclass_recall_metrics,
    multilabel_recall_metrics,
)
from arkas.metric.utils import check_label_type, check_same_shape_pred
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import numpy as np


class RecallResult(BaseResult):
    r"""Implement the recall result.

    This result can be used in 3 different settings:

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
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        y_pred: The predicted labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import RecallResult
    >>> # binary
    >>> result = RecallResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_pred=np.array([1, 0, 0, 1, 1]),
    ...     label_type="binary",
    ... )
    >>> result
    RecallResult(y_true=(5,), y_pred=(5,), label_type=binary)
    >>> result.compute_metrics()
    {'count': 5, 'recall': 1.0}
    >>> # multilabel
    >>> result = RecallResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ...     label_type="multilabel",
    ... )
    >>> result
    RecallResult(y_true=(5, 3), y_pred=(5, 3), label_type=multilabel)
    >>> result.compute_metrics()
    {'count': 5,
     'macro_recall': 0.666...,
     'micro_recall': 0.625,
     'recall': array([1., 1., 0.]),
     'weighted_recall': 0.625}
    >>> # multiclass
    >>> result = RecallResult(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ...     label_type="multiclass",
    ... )
    >>> result
    RecallResult(y_true=(6,), y_pred=(6,), label_type=multiclass)
    >>> result.compute_metrics()
    {'count': 6,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}
    >>> # auto
    >>> result = RecallResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    RecallResult(y_true=(5,), y_pred=(5,), label_type=binary)
    >>> result.compute_metrics()
    {'count': 5, 'recall': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, label_type: str = "auto") -> None:
        self._y_true = y_true
        self._y_pred = y_pred
        self._label_type = (
            find_label_type(y_true=y_true, y_pred=y_pred) if label_type == "auto" else label_type
        )

        self._check_inputs()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_pred={self._y_pred.shape}, label_type={self._label_type})"
        )

    @property
    def label_type(self) -> str:
        return self._label_type

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return recall_metrics(
            y_true=self._y_true,
            y_pred=self._y_pred,
            label_type=self._label_type,
            prefix=prefix,
            suffix=suffix,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)
            and self.label_type == other.label_type
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.ndim not in {1, 2}:
            msg = (
                f"'y_true' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_true.shape}"
            )
            raise ValueError(msg)
        if self._y_pred.ndim not in {1, 2}:
            msg = (
                f"'y_pred' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)
        check_label_type(self._label_type)


class BaseRecallResult(BaseResult):
    r"""Implement the base class to implement the recall results.

    Args:
        y_true: The ground truth target labels.
        y_pred: The predicted labels.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import BinaryRecallResult
    >>> result = BinaryRecallResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    BinaryRecallResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'count': 5, 'recall': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self._y_true = y_true
        self._y_pred = y_pred

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_pred={self._y_pred.shape})"
        )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.y_true, other.y_true, equal_nan=equal_nan
        ) and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)


class BinaryRecallResult(BaseRecallResult):
    r"""Implement the recall result for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, *)`` with ``0`` and
            ``1`` values.
        y_pred: The predicted labels. This input must be an array of
            shape ``(n_samples, *)`` with ``0`` and ``1`` values.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import BinaryRecallResult
    >>> result = BinaryRecallResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    BinaryRecallResult(y_true=(5,), y_pred=(5,))
    >>> result.compute_metrics()
    {'count': 5, 'recall': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        check_same_shape_pred(y_true, y_pred)
        super().__init__(y_true=y_true.ravel(), y_pred=y_pred.ravel())

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return binary_recall_metrics(
            y_true=self._y_true,
            y_pred=self._y_pred,
            prefix=prefix,
            suffix=suffix,
        )

    def generate_figures(self, prefix: str = "", suffix: str = "") -> dict[str, plt.Figure]:
        fig = binary_precision_recall_curve(
            y_true=self._y_true,
            y_pred=self._y_pred,
        )
        if fig is None:
            return {}
        return {f"{prefix}precision_recall{suffix}": fig}


class MulticlassRecallResult(BaseRecallResult):
    r"""Implement the recall result for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, *)`` with values in
            ``{0, ..., n_classes-1}``.
        y_pred: The predicted labels. This input must be an array of
            shape ``(n_samples, *)`` with values in
            ``{0, ..., n_classes-1}``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import MulticlassRecallResult
    >>> result = MulticlassRecallResult(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_pred=np.array([0, 0, 1, 1, 2, 2]),
    ... )
    >>> result
    MulticlassRecallResult(y_true=(6,), y_pred=(6,))
    >>> result.compute_metrics()
    {'count': 6,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        check_same_shape_pred(y_true, y_pred)
        super().__init__(y_true=y_true.ravel(), y_pred=y_pred.ravel())

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return multiclass_recall_metrics(
            y_true=self._y_true,
            y_pred=self._y_pred,
            prefix=prefix,
            suffix=suffix,
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}


class MultilabelRecallResult(BaseRecallResult):
    r"""Implement the recall result for multilabel labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, n_classes)`` with ``0``
            and ``1`` values.
        y_pred: The predicted labels. This input must be an array of
            shape ``(n_samples, n_classes)`` with ``0`` and ``1``
            values.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import MultilabelRecallResult
    >>> result = MultilabelRecallResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ... )
    >>> result
    MultilabelRecallResult(y_true=(5, 3), y_pred=(5, 3))
    >>> result.compute_metrics()
    {'count': 5,
     'macro_recall': 1.0,
     'micro_recall': 1.0,
     'recall': array([1., 1., 1.]),
     'weighted_recall': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        check_same_shape_pred(y_true, y_pred)
        super().__init__(y_true=y_true, y_pred=y_pred)

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return multilabel_recall_metrics(
            y_true=self._y_true,
            y_pred=self._y_pred,
            prefix=prefix,
            suffix=suffix,
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}