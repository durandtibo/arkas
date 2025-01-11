from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import BalancedAccuracyResult

############################################
#     Tests for BalancedAccuracyResult     #
############################################


def test_balanced_accuracy_result_y_true() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_balanced_accuracy_result_y_true_2d() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_balanced_accuracy_result_y_pred() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_balanced_accuracy_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_pred,
        np.array([1, 0, 1, 0, 1, 0]),
    )


def test_balanced_accuracy_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0])
        )


def test_balanced_accuracy_result_nan_policy() -> None:
    assert (
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]), nan_policy="omit"
        ).nan_policy
        == "omit"
    )


def test_balanced_accuracy_result_nan_policy_default() -> None:
    assert (
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).nan_policy
        == "propagate"
    )


def test_balanced_accuracy_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="incorrect",
        )


def test_balanced_accuracy_result_repr() -> None:
    assert repr(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BalancedAccuracyResult(")


def test_balanced_accuracy_result_str() -> None:
    assert str(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BalancedAccuracyResult(")


def test_balanced_accuracy_result_equal_true() -> None:
    assert BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_balanced_accuracy_result_equal_false_different_y_true() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_balanced_accuracy_result_equal_false_different_y_pred() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0]))
    )


def test_balanced_accuracy_result_equal_false_different_nan_policy() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), nan_policy="omit"
        )
    )


def test_balanced_accuracy_result_equal_false_different_type() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_balanced_accuracy_result_equal_nan_true() -> None:
    assert BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_balanced_accuracy_result_equal_nan_false() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_balanced_accuracy_result_compute_metrics_correct() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_result_compute_metrics_incorrect() -> None:
    result = BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 0.0, "count": 4},
    )


def test_balanced_accuracy_result_compute_metrics_empty() -> None:
    result = BalancedAccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": float("nan"), "count": 0},
        equal_nan=True,
    )


def test_balanced_accuracy_result_compute_metrics_prefix_suffix() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_balanced_accuracy_suffix": 1.0, "prefix_count_suffix": 5},
    )


def test_balanced_accuracy_result_compute_metrics_nan_omit() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_result_compute_metrics_nan_propagate() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


def test_balanced_accuracy_result_compute_metrics_nan_raise() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_balanced_accuracy_result_compute_metrics_binary() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1, 1])
    )
    assert objects_are_allclose(result.compute_metrics(), {"balanced_accuracy": 0.75, "count": 6})


def test_balanced_accuracy_result_compute_metrics_multiclass() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([0, 1, 2, 3, 4]), y_pred=np.array([0, 1, 1, 3, 3])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 0.6, "count": 5},
    )


def test_balanced_accuracy_result_generate_figures() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_balanced_accuracy_result_generate_figures_empty() -> None:
    result = BalancedAccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
