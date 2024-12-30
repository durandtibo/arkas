from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import balanced_accuracy

#######################################
#     Tests for balanced_accuracy     #
#######################################


def test_balanced_accuracy_binary_correct() -> None:
    assert objects_are_equal(
        balanced_accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_binary_correct_2d() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]), y_pred=np.array([[1, 0, 0], [1, 1, 0]])
        ),
        {"balanced_accuracy": 1.0, "count": 6},
    )


def test_balanced_accuracy_binary_incorrect() -> None:
    assert objects_are_equal(
        balanced_accuracy(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0])),
        {"balanced_accuracy": 0.0, "count": 4},
    )


def test_balanced_accuracy_multiclass_correct() -> None:
    assert objects_are_equal(
        balanced_accuracy(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])),
        {"balanced_accuracy": 1.0, "count": 6},
    )


def test_balanced_accuracy_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        balanced_accuracy(
            y_true=np.array([0, 0, 1, 1, 2, 2, 3]), y_pred=np.array([0, 0, 1, 1, 1, 1, 3])
        ),
        {"balanced_accuracy": 0.75, "count": 7},
    )


def test_balanced_accuracy_empty() -> None:
    assert objects_are_equal(
        balanced_accuracy(y_true=np.array([]), y_pred=np.array([])),
        {"balanced_accuracy": float("nan"), "count": 0},
        equal_nan=True,
    )


def test_balanced_accuracy_prefix_suffix() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_balanced_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
        },
    )


def test_balanced_accuracy_nan_omit() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="omit",
        ),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_nan_omit_y_true() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="omit",
        ),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="omit",
        ),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_nan_propagate() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="propagate",
        ),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


def test_balanced_accuracy_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="propagate",
        ),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


def test_balanced_accuracy_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="propagate",
        ),
        {"balanced_accuracy": float("nan"), "count": 6},
        equal_nan=True,
    )


def test_balanced_accuracy_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="raise",
        )


def test_balanced_accuracy_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="raise",
        )


def test_balanced_accuracy_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        balanced_accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="raise",
        )
