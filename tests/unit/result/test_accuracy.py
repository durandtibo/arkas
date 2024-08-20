from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.result import AccuracyResult

####################################
#     Tests for AccuracyResult     #
####################################


def test_accuracy_result_y_true() -> None:
    assert objects_are_equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_accuracy_result_y_pred() -> None:
    assert objects_are_equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_accuracy_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_pred' have different shapes"):
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0]))


def test_accuracy_result_repr() -> None:
    assert repr(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("AccuracyResult(")


def test_accuracy_result_str() -> None:
    assert str(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("AccuracyResult(")


def test_accuracy_result_equal_true() -> None:
    assert AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])).equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_accuracy_result_equal_false_different_y_true() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(AccuracyResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])))


def test_accuracy_result_equal_false_different_y_pred() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_accuracy_result_equal_false_different_type() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_accuracy_result_compute_metrics_correct() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_result_compute_metrics_incorrect() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_result_compute_metrics_empty() -> None:
    result = AccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": 0,
            "count_incorrect": 0,
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_result_compute_metrics_prefix_suffix() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


def test_accuracy_result_generate_figures() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_accuracy_result_generate_figures_empty() -> None:
    result = AccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
