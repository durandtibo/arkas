r"""Implement the mean absolute error result."""

from __future__ import annotations

__all__ = ["MeanTweedieDevianceResult"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils.format import repr_mapping_line

from arkas.metric.regression.tweedie_deviance import mean_tweedie_deviance
from arkas.metric.utils import check_nan_policy, check_same_shape_pred
from arkas.result.base import BaseResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


class MeanTweedieDevianceResult(BaseResult):
    r"""Implement the mean Tweedie deviance regression loss result.

    Args:
        y_true: The ground truth target values.
        y_pred: The predicted values.
        powers: The Tweedie power parameter. The higher power the less
            weight is given to extreme deviations between true and
            predicted targets.
        nan_policy: The policy on how to handle NaN values in the input
            arrays. The following options are available: ``'omit'``,
            ``'propagate'``, and ``'raise'``.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.result import MeanTweedieDevianceResult
    >>> result = MeanTweedieDevianceResult(
    ...     y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
    ... )
    >>> result
    MeanTweedieDevianceResult(y_true=(5,), y_pred=(5,), powers=(0,), nan_policy='propagate')
    >>> result.compute_metrics()
    {'count': 5, 'mean_tweedie_deviance_power_0': 0.0}

    ```
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        powers: Sequence[float] = (0,),
        nan_policy: str = "propagate",
    ) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()
        check_same_shape_pred(y_true=self._y_true, y_pred=self._y_pred)

        self._powers = tuple(powers)

        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

    def __repr__(self) -> str:
        args = repr_mapping_line(
            {
                "y_true": self._y_true.shape,
                "y_pred": self._y_pred.shape,
                "powers": self._powers,
                "nan_policy": self._nan_policy,
            }
        )
        return f"{self.__class__.__qualname__}({args})"

    @property
    def nan_policy(self) -> str:
        return self._nan_policy

    @property
    def powers(self) -> tuple[float, ...]:
        return self._powers

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return mean_tweedie_deviance(
            y_true=self._y_true,
            y_pred=self._y_pred,
            prefix=prefix,
            suffix=suffix,
            powers=self._powers,
            nan_policy=self._nan_policy,
        )

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            objects_are_equal(self.y_true, other.y_true, equal_nan=equal_nan)
            and objects_are_equal(self.y_pred, other.y_pred, equal_nan=equal_nan)
            and objects_are_equal(self.powers, other.powers)
            and self.nan_policy == other.nan_policy
        )

    def generate_figures(
        self, prefix: str = "", suffix: str = ""  # noqa: ARG002
    ) -> dict[str, float]:
        return {}