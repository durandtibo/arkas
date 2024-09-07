from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric import mean_squared_error

########################################
#     Tests for mean_squared_error     #
########################################


def test_mean_squared_error_correct() -> None:
    assert objects_are_equal(
        mean_squared_error(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "mean_squared_error": 0.0},
    )


def test_mean_squared_error_correct_2d() -> None:
    assert objects_are_equal(
        mean_squared_error(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "mean_squared_error": 0.0},
    )


def test_mean_squared_error_incorrect() -> None:
    assert objects_are_equal(
        mean_squared_error(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "mean_squared_error": 5.0},
    )


def test_mean_squared_error_empty() -> None:
    assert objects_are_equal(
        mean_squared_error(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "mean_squared_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_error_prefix_suffix() -> None:
    assert objects_are_equal(
        mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_mean_squared_error_suffix": 0.0},
    )


def test_mean_squared_error_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        mean_squared_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
        )


def test_mean_squared_error_ignore_nan() -> None:
    assert objects_are_equal(
        mean_squared_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            ignore_nan=True,
        ),
        {"count": 3, "mean_squared_error": 0.0},
    )


def test_mean_squared_error_ignore_nan_y_true() -> None:
    assert objects_are_equal(
        mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            ignore_nan=True,
        ),
        {"count": 5, "mean_squared_error": 0.0},
    )


def test_mean_squared_error_ignore_nan_y_pred() -> None:
    assert objects_are_equal(
        mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            ignore_nan=True,
        ),
        {"count": 5, "mean_squared_error": 0.0},
    )
