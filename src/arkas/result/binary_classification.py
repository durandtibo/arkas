r"""Implement the default binary classification result."""

from __future__ import annotations

__all__ = ["BinaryClassificationResult"]

from typing import TYPE_CHECKING

import numpy as np
from sklearn import metrics

from arkas.result.base import BaseResult

if TYPE_CHECKING:
    from collections.abc import Sequence


class BinaryClassificationResult(BaseResult):
    r"""Implement the default binary classification result.

    Args:
        y_true: The ground truth target binary labels. This input must
            be an array of shape ``(n_samples,)`` where the values
            are ``0`` or ``1``.
        y_pred: The predicted binary labels. This input must be an
            array of shape ``(n_samples,)`` where the values are ``0``
            or ``1``.
        y_score: The target scores, can either be probability
            estimates of the positive class, confidence values,
            or non-thresholded measure of decisions.
        f1_betas: The betas used to compute the F-beta scores.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray | None = None,
        f1_betas: Sequence[float] = (1,),
    ) -> None:
        self._y_true = y_true.ravel()
        self._y_pred = y_pred.ravel()
        self._y_score = None if y_score is None else y_score.ravel().astype(np.float64)
        self._f1_betas = tuple(f1_betas)

        self._check_inputs()

    @property
    def y_true(self) -> np.ndarray:
        return self._y_true

    @property
    def y_pred(self) -> np.ndarray:
        return self._y_pred

    @property
    def y_score(self) -> np.ndarray | None:
        return self._y_score

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return (
            self.compute_base_metrics(prefix=prefix, suffix=suffix)
            | self.compute_confmat_metrics(prefix=prefix, suffix=suffix)
            | self.compute_fbeta_metrics(prefix=prefix, suffix=suffix)
            | self.compute_rank_metrics(prefix=prefix, suffix=suffix)
        )

    def compute_base_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return {
            f"{prefix}accuracy{suffix}": float(
                metrics.accuracy_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
            f"{prefix}balanced_accuracy{suffix}": float(
                metrics.balanced_accuracy_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
            f"{prefix}precision{suffix}": float(
                metrics.precision_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
            f"{prefix}recall{suffix}": float(
                metrics.recall_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
            f"{prefix}jaccard{suffix}": float(
                metrics.jaccard_score(y_true=self._y_true, y_pred=self._y_pred)
            ),
        }

    def compute_confmat_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        count = self._y_true.size
        count_correct = int(
            metrics.accuracy_score(y_true=self._y_true, y_pred=self._y_pred, normalize=False)
        )
        return {
            f"{prefix}count{suffix}": count,
            f"{prefix}count_correct{suffix}": count_correct,
            f"{prefix}count_incorrect{suffix}": count - count_correct,
        }

    def compute_fbeta_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return {
            f"{prefix}f{beta}{suffix}": float(
                metrics.fbeta_score(y_true=self._y_true, y_pred=self._y_pred, beta=beta)
            )
            for beta in self._f1_betas
        }

    def compute_rank_metrics(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        if self._y_score is None:
            return {}
        return {
            f"{prefix}average_precision{suffix}": float(
                metrics.average_precision_score(y_true=self._y_true, y_score=self._y_score)
            ),
            f"{prefix}roc_auc{suffix}": float(
                metrics.roc_auc_score(y_true=self._y_true, y_score=self._y_score)
            ),
        }

    def generate_plots(self, prefix: str = "", suffix: str = "") -> dict[str, float]:
        return {}

    def _check_inputs(self) -> None:
        if self._y_true.shape != self._y_pred.shape:
            msg = (
                f"'y_true' and 'y_pred' have different shapes: {self._y_true.shape} vs "
                f"{self._y_pred.shape}"
            )
            raise ValueError(msg)
        if self._y_score is not None and self._y_true.shape != self._y_score.shape:
            msg = (
                f"'y_true' and 'y_score' have different shapes: {self._y_true.shape} vs "
                f"{self._y_score.shape}"
            )
            raise ValueError(msg)
