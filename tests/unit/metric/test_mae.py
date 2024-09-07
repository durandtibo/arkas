from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from arkas.metric import mean_absolute_error

#########################################
#     Tests for mean_absolute_error     #
#########################################


def test_mean_absolute_error_correct() -> None:
    assert objects_are_equal(
        mean_absolute_error(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "mean_absolute_error": 0.0},
    )


def test_mean_absolute_error_correct_2d() -> None:
    assert objects_are_equal(
        mean_absolute_error(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "mean_absolute_error": 0.0},
    )


def test_mean_absolute_error_incorrect() -> None:
    assert objects_are_equal(
        mean_absolute_error(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "mean_absolute_error": 2.0},
    )


def test_mean_absolute_error_empty() -> None:
    assert objects_are_equal(
        mean_absolute_error(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "mean_absolute_error": float("nan")},
        equal_nan=True,
    )


def test_mean_absolute_error_prefix_suffix() -> None:
    assert objects_are_equal(
        mean_absolute_error(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_mean_absolute_error_suffix": 0.0},
    )


def test_mean_absolute_error_nans() -> None:
    assert objects_are_equal(
        mean_absolute_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
        ),
        {"count": 3, "mean_absolute_error": 0.0},
    )


def test_mean_absolute_error_y_true_nan() -> None:
    assert objects_are_equal(
        mean_absolute_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
        ),
        {"count": 5, "mean_absolute_error": 0.0},
    )


def test_mean_absolute_error_y_pred_nan() -> None:
    assert objects_are_equal(
        mean_absolute_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
        ),
        {"count": 5, "mean_absolute_error": 0.0},
    )
