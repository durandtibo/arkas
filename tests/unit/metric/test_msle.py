from __future__ import annotations

import numpy as np
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import mean_squared_log_error_metrics

####################################################
#     Tests for mean_squared_log_error_metrics     #
####################################################


def test_mean_squared_log_error_metrics_correct() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
        ),
        {"count": 5, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_metrics_correct_2d() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_metrics_incorrect() -> None:
    assert objects_are_allclose(
        mean_squared_log_error_metrics(
            y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])
        ),
        {"count": 4, "mean_squared_log_error": 0.46117484006431314},
    )


def test_mean_squared_log_error_metrics_empty() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "mean_squared_log_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_log_error_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_mean_squared_log_error_suffix": 0.0},
    )


def test_mean_squared_log_error_metrics_nans() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
        ),
        {"count": 3, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
        ),
        {"count": 5, "mean_squared_log_error": 0.0},
    )


def test_mean_squared_log_error_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        mean_squared_log_error_metrics(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
        ),
        {"count": 5, "mean_squared_log_error": 0.0},
    )
