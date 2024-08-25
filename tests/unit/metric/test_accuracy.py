from __future__ import annotations

import numpy as np
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import accuracy_metrics

######################################
#     Tests for accuracy_metrics     #
######################################


def test_accuracy_metrics_binary_correct() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]), y_pred=np.array([[1, 0, 0], [1, 1, 0]])
        ),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0])),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_metrics_multiclass_correct() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        accuracy_metrics(y_true=np.array([0, 0, 1, 1, 2]), y_pred=np.array([0, 0, 1, 1, 1])),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


def test_accuracy_metrics_empty() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": 0,
            "count_incorrect": 0,
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


def test_accuracy_metrics_nans() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )
