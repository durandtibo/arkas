from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import root_mean_squared_error
from tests.conftest import sklearn_greater_equal_1_4

#############################################
#     Tests for root_mean_squared_error     #
#############################################


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_correct() -> None:
    assert objects_are_equal(
        root_mean_squared_error(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_correct_2d() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_incorrect() -> None:
    assert objects_are_allclose(
        root_mean_squared_error(y_true=np.array([1, 2, 3, 4]), y_pred=np.array([3, 5, 4, 5])),
        {"count": 4, "root_mean_squared_error": 1.9364916731037085},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_empty() -> None:
    assert objects_are_equal(
        root_mean_squared_error(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_prefix_suffix() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_root_mean_squared_error_suffix": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_omit() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 3, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_omit_y_true() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_propagate() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {"count": 6, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


def test_root_mean_squared_error_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        root_mean_squared_error(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


def test_root_mean_squared_error_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="raise",
        )


def test_root_mean_squared_error_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        root_mean_squared_error(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="raise",
        )
