from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from arkas.plot import binary_precision_recall_curve

###################################################
#     Tests for binary_precision_recall_curve     #
###################################################


def test_binary_precision_recall_curve() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax, y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )


def test_binary_precision_recall_curve_empty() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax, y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )


def test_binary_precision_recall_curve_nan() -> None:
    _fig, ax = plt.subplots()
    binary_precision_recall_curve(
        ax,
        y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan"), 1, float("nan")]),
    )
