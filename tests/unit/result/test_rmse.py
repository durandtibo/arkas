from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import RootMeanSquaredErrorResult
from tests.conftest import sklearn_greater_equal_1_4

################################################
#     Tests for RootMeanSquaredErrorResult     #
################################################


def test_root_mean_squared_error_result_y_true() -> None:
    assert objects_are_equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_true,
        np.array([1, 2, 3, 4, 5]),
    )


def test_root_mean_squared_error_result_y_true_2d() -> None:
    assert objects_are_equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_true,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_root_mean_squared_error_result_y_pred() -> None:
    assert objects_are_equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_pred,
        np.array([5, 4, 3, 2, 1]),
    )


def test_root_mean_squared_error_result_y_pred_2d() -> None:
    assert objects_are_equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_pred,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_root_mean_squared_error_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0])
        )


def test_root_mean_squared_error_result_nan_policy() -> None:
    assert (
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([5, 4, 3, 2, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_root_mean_squared_error_result_nan_policy_default() -> None:
    assert (
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).nan_policy
        == "propagate"
    )


def test_root_mean_squared_error_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([5, 4, 3, 2, 1]),
            nan_policy="incorrect",
        )


def test_root_mean_squared_error_result_repr() -> None:
    assert repr(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("RootMeanSquaredErrorResult(")


def test_root_mean_squared_error_result_str() -> None:
    assert str(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("RootMeanSquaredErrorResult(")


def test_root_mean_squared_error_result_equal_true() -> None:
    assert RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_root_mean_squared_error_result_equal_false_different_y_true() -> None:
    assert not RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_root_mean_squared_error_result_equal_false_different_y_pred() -> None:
    assert not RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])
        )
    )


def test_root_mean_squared_error_result_equal_false_different_nan_policy() -> None:
    assert not RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            nan_policy="omit",
        )
    )


def test_root_mean_squared_error_result_equal_false_different_type() -> None:
    assert not RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_root_mean_squared_error_result_equal_nan_true() -> None:
    assert RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_root_mean_squared_error_result_equal_nan_false() -> None:
    assert not RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        RootMeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_correct() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "root_mean_squared_error": 0.0})


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_incorrect() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([1, 2, 3, 4]), y_pred=np.array([3, 5, 4, 5])
    )
    assert objects_are_allclose(
        result.compute_metrics(), {"count": 4, "root_mean_squared_error": 1.9364916731037085}
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_empty() -> None:
    result = RootMeanSquaredErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 0, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_prefix_suffix() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_root_mean_squared_error_suffix": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_nan_omit() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 4, "root_mean_squared_error": 0.0},
    )


@sklearn_greater_equal_1_4
def test_root_mean_squared_error_result_compute_metrics_nan_propagate() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 7, "root_mean_squared_error": float("nan")},
        equal_nan=True,
    )


def test_root_mean_squared_error_result_compute_metrics_nan_raise() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_root_mean_squared_error_result_generate_figures() -> None:
    result = RootMeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_root_mean_squared_error_result_generate_figures_empty() -> None:
    result = RootMeanSquaredErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
