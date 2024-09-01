r"""Implement the average precision result."""

from __future__ import annotations

__all__ = [
    "AveragePrecisionResult",
    "BinaryAveragePrecisionResult",
    "BaseAveragePrecisionResult",
    "MulticlassAveragePrecisionResult",
]

from typing import TYPE_CHECKING, Any

import numpy as np
from coola import objects_are_equal

from arkas.metric import binary_average_precision_metrics
from arkas.metric.ap import (
    average_precision_metrics,
    find_label_type,
    multiclass_average_precision_metrics,
)
from arkas.metric.utils import check_label_type, check_same_shape_score
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


class AveragePrecisionResult(BaseResult):
    r"""Implement the average precision result.

    This result can be used in 3 different settings:

    - binary: ``y_true`` must be an array of shape ``(*)``
        with ``0`` and ``1`` values, and ``y_score`` must be an array
        of shape ``(*)``.
    - multiclass: ``y_true`` must be an array of shape ``(n_samples,)``
        with values in ``{0, ..., n_classes-1}``, and ``y_score`` must
        be an array of shape ``(n_samples, n_classes)``.
    - multilabel: ``y_true`` must be an array of shape
        ``(n_samples, n_classes)`` with ``0`` and ``1`` values, and
        ``y_score`` must be an array of shape
        ``(n_samples, n_classes)``.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples,)`` or
            ``(n_samples, n_classes)``.
        label_type: The type of labels used to evaluate the metrics.
            The valid values are: ``'binary'``, ``'multiclass'``,
            ``'multilabel'``, and ``'auto'``. If ``'binary'`` or
            ``'multilabel'``, ``y_true`` values  must be ``0`` and
            ``1``. If ``'auto'``, it tries to automatically find the
            label type from the arrays' shape.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import AveragePrecisionResult
    >>> # binary
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_score=np.array([2, -1, 0, 3, 1]),
    ...     label_type="binary",
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5,), y_score=(5,), label_type=binary)
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}
    >>> # multilabel
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    ...     y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
    ...     label_type="multilabel",
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5, 3), y_score=(5, 3), label_type=multilabel)
    >>> result.compute_metrics()
    {'average_precision': array([1. , 1. , 0.477...]),
     'count': 5,
     'macro_average_precision': 0.825...,
     'micro_average_precision': 0.588...,
     'weighted_average_precision': 0.804...}
    >>> # multiclass
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_score=np.array(
    ...         [
    ...             [0.7, 0.2, 0.1],
    ...             [0.4, 0.3, 0.3],
    ...             [0.1, 0.8, 0.1],
    ...             [0.2, 0.3, 0.5],
    ...             [0.4, 0.4, 0.2],
    ...             [0.1, 0.2, 0.7],
    ...         ]
    ...     ),
    ...     label_type="multiclass",
    ... )
    >>> result
    AveragePrecisionResult(y_true=(6,), y_score=(6, 3), label_type=multiclass)
    >>> result.compute_metrics()
    {'average_precision': array([0.833..., 0.75 , 0.75 ]),
     'count': 6,
     'macro_average_precision': 0.777...,
     'micro_average_precision': 0.75,
     'weighted_average_precision': 0.777...}
    >>> # auto
    >>> result = AveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]),
    ...     y_score=np.array([2, -1, 0, 3, 1]),
    ... )
    >>> result
    AveragePrecisionResult(y_true=(5,), y_score=(5,), label_type=binary)
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray, label_type: str = "auto") -> None:
        self._y_true = y_true
        self._y_score = y_score.astype(np.float64)
        self._label_type = (
            find_label_type(y_true=y_true, y_score=y_score) if label_type == "auto" else label_type
        )

        self._check_inputs()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_score={self._y_score.shape}, label_type={self._label_type})"
        )

    @property
    def label_type(self) -> str:
        return self._label_type

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_score(self) -> np.ndarray:
        return self._y_score

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return average_precision_metrics(
            y_true=self._y_true,
            y_score=self._y_score,
            label_type=self._label_type,
            prefix=prefix,
            suffix=suffix,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_score, other.y_score, equal_nan=equal_nan)
            and self.label_type == other.label_type
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, plt.Figure]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.ndim not in {1, 2}:
            msg = (
                f"'y_true' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_true.shape}"
            )
            raise ValueError(msg)
        if self._y_score.ndim not in {1, 2}:
            msg = (
                f"'y_score' must be a 1d or 2d array but received an array of shape: "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
        if self._y_true.ndim == self._y_score.ndim and self._y_true.shape != self._y_score.shape:
            msg = (
                f"'y_true' and 'y_score' have different shapes: {self._y_true.shape} vs "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
        check_label_type(self._label_type)


class BaseAveragePrecisionResult(BaseResult):
    r"""Implement the base class to implement the precision results.

    Args:
        y_true: The ground truth target labels.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import BinaryAveragePrecisionResult
    >>> result = BinaryAveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 1])
    ... )
    >>> result
    BinaryAveragePrecisionResult(y_true=(5,), y_score=(5,))
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        self._y_true = y_true
        self._y_score = y_score

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__qualname__}(y_true={self._y_true.shape}, "
            f"y_score={self._y_score.shape})"
        )

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_score(self) -> np.ndarray:
        return self._y_score

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(
            self.y_true, other.y_true, equal_nan=equal_nan
        ) and objects_are_equal(self.y_score, other.y_score, equal_nan=equal_nan)


class BinaryAveragePrecisionResult(BaseAveragePrecisionResult):
    r"""Implement the precision result for binary labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, *)`` with ``0`` and
            ``1`` values.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples, *)``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import BinaryAveragePrecisionResult
    >>> result = BinaryAveragePrecisionResult(
    ...     y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ... )
    >>> result
    BinaryAveragePrecisionResult(y_true=(5,), y_score=(5,))
    >>> result.compute_metrics()
    {'average_precision': 1.0, 'count': 5}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        check_same_shape_score(y_true, y_score)
        super().__init__(y_true=y_true.ravel(), y_score=y_score.ravel())

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return binary_average_precision_metrics(
            y_true=self._y_true,
            y_score=self._y_score,
            prefix=prefix,
            suffix=suffix,
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, plt.Figure]:
        return {}


class MulticlassAveragePrecisionResult(BaseAveragePrecisionResult):
    r"""Implement the precision result for multiclass labels.

    Args:
        y_true: The ground truth target labels. This input must
            be an array of shape ``(n_samples, *)`` with ``0`` and
            ``1`` values.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions. This input must
            be an array of shape ``(n_samples, *)``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import MulticlassAveragePrecisionResult
    >>> result = MulticlassAveragePrecisionResult(
    ...     y_true=np.array([0, 0, 1, 1, 2, 2]),
    ...     y_score=np.array(
    ...         [
    ...             [0.7, 0.2, 0.1],
    ...             [0.4, 0.3, 0.3],
    ...             [0.1, 0.8, 0.1],
    ...             [0.2, 0.5, 0.3],
    ...             [0.3, 0.3, 0.4],
    ...             [0.1, 0.2, 0.7],
    ...         ]
    ...     ),
    ... )
    >>> result
    MulticlassAveragePrecisionResult(y_true=(6,), y_score=(6, 3))
    >>> result.compute_metrics()
    {'average_precision': array([1., 1., 1.]),
     'count': 6,
     'macro_average_precision': 1.0,
     'micro_average_precision': 1.0,
     'weighted_average_precision': 1.0}

    ```
    """

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray) -> None:
        y_true = y_true.ravel()
        if y_true.shape[0] != y_score.shape[0]:
            msg = (
                f"'y_true' and 'y_score' have different first dimension: {y_true.shape} vs "
                f"{y_score.shape}"
            )
            raise RuntimeError(msg)
        super().__init__(y_true=y_true, y_score=y_score)

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return multiclass_average_precision_metrics(
            y_true=self._y_true,
            y_score=self._y_score,
            prefix=prefix,
            suffix=suffix,
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, plt.Figure]:
        return {}
