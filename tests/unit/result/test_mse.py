from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.result import MeanSquaredErrorResult

############################################
#     Tests for MeanSquaredErrorResult     #
############################################


def test_mean_squared_error_result_y_true() -> None:
    assert objects_are_equal(
        MeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_true,
        np.array([1, 2, 3, 4, 5]),
    )


def test_mean_squared_error_result_y_true_2d() -> None:
    assert objects_are_equal(
        MeanSquaredErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_true,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_mean_squared_error_result_y_pred() -> None:
    assert objects_are_equal(
        MeanSquaredErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_pred,
        np.array([5, 4, 3, 2, 1]),
    )


def test_mean_squared_error_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MeanSquaredErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_pred,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_mean_squared_error_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0])
        )


def test_mean_squared_error_result_repr() -> None:
    assert repr(
        MeanSquaredErrorResult(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]))
    ).startswith("MeanSquaredErrorResult(")


def test_mean_squared_error_result_str() -> None:
    assert str(
        MeanSquaredErrorResult(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]))
    ).startswith("MeanSquaredErrorResult(")


def test_mean_squared_error_result_equal_true() -> None:
    assert MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanSquaredErrorResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_mean_squared_error_result_equal_false_different_y_true() -> None:
    assert not MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanSquaredErrorResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_mean_squared_error_result_equal_false_different_y_pred() -> None:
    assert not MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanSquaredErrorResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0]))
    )


def test_mean_squared_error_result_equal_false_different_type() -> None:
    assert not MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_mean_squared_error_result_equal_nan_true() -> None:
    assert MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        MeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_mean_squared_error_result_equal_nan_false() -> None:
    assert not MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        MeanSquaredErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_mean_squared_error_result_compute_metrics_correct() -> None:
    result = MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "mean_squared_error": 0.0})


def test_mean_squared_error_result_compute_metrics_incorrect() -> None:
    result = MeanSquaredErrorResult(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4]))
    assert objects_are_equal(result.compute_metrics(), {"count": 4, "mean_squared_error": 5.0})


def test_mean_squared_error_result_compute_metrics_empty() -> None:
    result = MeanSquaredErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 0, "mean_squared_error": float("nan")},
        equal_nan=True,
    )


def test_mean_squared_error_result_compute_metrics_prefix_suffix() -> None:
    result = MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_mean_squared_error_suffix": 0.0},
    )


def test_mean_squared_error_result_generate_figures() -> None:
    result = MeanSquaredErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_mean_squared_error_result_generate_figures_empty() -> None:
    result = MeanSquaredErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
