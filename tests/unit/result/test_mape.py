from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import MeanAbsolutePercentageErrorResult

#######################################################
#     Tests for MeanAbsolutePercentageErrorResult     #
#######################################################


def test_mean_absolute_percentage_error_result_y_true() -> None:
    assert objects_are_equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_true,
        np.array([1, 2, 3, 4, 5]),
    )


def test_mean_absolute_percentage_error_result_y_true_2d() -> None:
    assert objects_are_equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_true,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_mean_absolute_percentage_error_result_y_pred() -> None:
    assert objects_are_equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).y_pred,
        np.array([5, 4, 3, 2, 1]),
    )


def test_mean_absolute_percentage_error_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[6, 5, 4], [3, 2, 1]])
        ).y_pred,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_mean_absolute_percentage_error_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0])
        )


def test_mean_absolute_percentage_error_result_nan_policy() -> None:
    assert (
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([5, 4, 3, 2, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_mean_absolute_percentage_error_result_nan_policy_default() -> None:
    assert (
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([5, 4, 3, 2, 1])
        ).nan_policy
        == "propagate"
    )


def test_mean_absolute_percentage_error_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([5, 4, 3, 2, 1]),
            nan_policy="incorrect",
        )


def test_mean_absolute_percentage_error_result_repr() -> None:
    assert repr(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("MeanAbsolutePercentageErrorResult(")


def test_mean_absolute_percentage_error_result_str() -> None:
    assert str(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("MeanAbsolutePercentageErrorResult(")


def test_mean_absolute_percentage_error_result_equal_true() -> None:
    assert MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_mean_absolute_percentage_error_result_equal_false_different_y_true() -> None:
    assert not MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_mean_absolute_percentage_error_result_equal_false_different_y_pred() -> None:
    assert not MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])
        )
    )


def test_mean_absolute_percentage_error_result_equal_false_different_nan_policy() -> None:
    assert not MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            nan_policy="omit",
        )
    )


def test_mean_absolute_percentage_error_result_equal_false_different_type() -> None:
    assert not MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_mean_absolute_percentage_error_result_equal_nan_true() -> None:
    assert MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_mean_absolute_percentage_error_result_equal_nan_false() -> None:
    assert not MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        MeanAbsolutePercentageErrorResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_mean_absolute_percentage_error_result_compute_metrics_correct() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(), {"count": 5, "mean_absolute_percentage_error": 0.0}
    )


def test_mean_absolute_percentage_error_result_compute_metrics_incorrect() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])
    )
    assert objects_are_allclose(
        result.compute_metrics(), {"count": 4, "mean_absolute_percentage_error": 1.1458333333333333}
    )


def test_mean_absolute_percentage_error_result_compute_metrics_empty() -> None:
    result = MeanAbsolutePercentageErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 0, "mean_absolute_percentage_error": float("nan")},
        equal_nan=True,
    )


def test_mean_absolute_percentage_error_result_compute_metrics_prefix_suffix() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_mean_absolute_percentage_error_suffix": 0.0},
    )


def test_mean_absolute_percentage_error_result_compute_metrics_nan_omit() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(), {"count": 4, "mean_absolute_percentage_error": 0.0}
    )


def test_mean_absolute_percentage_error_result_compute_metrics_nan_propagate() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 7, "mean_absolute_percentage_error": float("nan")},
        equal_nan=True,
    )


def test_mean_absolute_percentage_error_result_compute_metrics_nan_raise() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
        y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_mean_absolute_percentage_error_result_generate_figures() -> None:
    result = MeanAbsolutePercentageErrorResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_mean_absolute_percentage_error_result_generate_figures_empty() -> None:
    result = MeanAbsolutePercentageErrorResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
