from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from arkas.plot import binary_roc_curve

######################################
#     Tests for binary_roc_curve     #
######################################


def test_binary_roc_curve() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(ax, y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))


def test_binary_roc_curve_empty() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(ax, y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))


def test_binary_roc_curve_nan() -> None:
    _fig, ax = plt.subplots()
    binary_roc_curve(
        ax,
        y_true=np.array([1, 0, 0, 1, 1, float("nan"), float("nan"), 1]),
        y_score=np.array([2, -1, 0, 3, 1, float("nan"), 1, float("nan")]),
    )
