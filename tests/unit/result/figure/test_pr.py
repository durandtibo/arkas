from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from arkas.result.figure import binary_precision_recall_curve

###################################################
#     Tests for binary_precision_recall_curve     #
###################################################


def test_binary_precision_recall_curve() -> None:
    assert isinstance(
        binary_precision_recall_curve(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ),
        plt.Figure,
    )


def test_binary_precision_recall_curve_empty() -> None:
    assert binary_precision_recall_curve(y_true=np.array([]), y_pred=np.array([])) is None


def test_binary_precision_recall_curve_nan() -> None:
    assert isinstance(
        binary_precision_recall_curve(
            y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan"), 1, float("nan")]),
        ),
        plt.Figure,
    )
