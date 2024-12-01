from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric import (
    binary_top_k_accuracy,
    multiclass_top_k_accuracy,
    top_k_accuracy,
)

####################################
#     Tests for top_k_accuracy     #
####################################


def test_top_k_accuracy_empty() -> None:
    assert objects_are_equal(
        top_k_accuracy(y_true=np.array([]), y_score=np.array([]), k=[1, 2]),
        {"count": 0, "top_1_accuracy": float("nan"), "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_top_k_accuracy_binary_correct() -> None:
    assert objects_are_equal(
        top_k_accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])),
        {"count": 5, "top_2_accuracy": 1.0},
    )


def test_top_k_accuracy_binary_k() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]), k=[1, 2]
        ),
        {"count": 5, "top_1_accuracy": 1.0, "top_2_accuracy": 1.0},
    )


def test_top_k_accuracy_binary_incorrect() -> None:
    assert objects_are_equal(
        top_k_accuracy(y_true=np.array([1, 0, 0, 1]), y_score=np.array([0, 1, 1, 0]), k=[1, 2]),
        {"count": 4, "top_1_accuracy": 0.0, "top_2_accuracy": 1.0},
    )


def test_top_k_accuracy_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            k=[1, 2],
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_top_1_accuracy_suffix": 1.0,
            "prefix_top_2_accuracy_suffix": 1.0,
        },
    )


def test_top_k_accuracy_multiclass() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
        ),
        {"count": 4, "top_2_accuracy": 0.75},
    )


def test_top_k_accuracy_multiclass_k() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            k=[1, 2, 3],
        ),
        {"count": 4, "top_1_accuracy": 0.5, "top_2_accuracy": 0.75, "top_3_accuracy": 1.0},
    )


def test_top_k_accuracy_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            k=[1, 2, 3],
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 4,
            "prefix_top_1_accuracy_suffix": 0.5,
            "prefix_top_2_accuracy_suffix": 0.75,
            "prefix_top_3_accuracy_suffix": 1.0,
        },
    )


def test_top_k_accuracy_binary_nan_omit() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_top_k_accuracy_nan_propagate() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
        ),
        {"count": 5, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_top_k_accuracy_binary_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="raise",
        )


def test_top_k_accuracy_multiclass_nan_omit() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan"), 0]),
            y_score=np.array(
                [
                    [float("nan"), 0.2, 0.2],
                    [0.3, 0.4, 0.2],
                    [0.2, 0.4, 0.3],
                    [0.7, 0.2, 0.1],
                    [0.5, 0.2, 0.2],
                ]
            ),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_top_k_accuracy_multiclass_nan_propagate() -> None:
    assert objects_are_equal(
        top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
        ),
        {"count": 4, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_top_k_accuracy_multiclass_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
            nan_policy="raise",
        )


###########################################
#     Tests for binary_top_k_accuracy     #
###########################################


def test_binary_top_k_accuracy_correct() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])),
        {"count": 5, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_incorrect() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1]), y_score=np.array([0, 1, 1, 0]), k=[1, 2]
        ),
        {"count": 4, "top_1_accuracy": 0.0, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_k() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]), k=[1, 2]
        ),
        {"count": 5, "top_1_accuracy": 1.0, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            k=[1, 2],
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_top_1_accuracy_suffix": 1.0,
            "prefix_top_2_accuracy_suffix": 1.0,
        },
    )


def test_binary_top_k_accuracy_nan_omit() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_nan_omit_y_true() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 4, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_nan_omit_y_score() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="omit",
        ),
        {"count": 4, "top_2_accuracy": 1.0},
    )


def test_binary_top_k_accuracy_nan_propagate() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
        ),
        {"count": 5, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_binary_top_k_accuracy_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {"count": 5, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_binary_top_k_accuracy_nan_propagate_y_score() -> None:
    assert objects_are_equal(
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
        ),
        {"count": 5, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_binary_top_k_accuracy_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="raise",
        )


def test_binary_top_k_accuracy_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1]),
            nan_policy="raise",
        )


def test_binary_top_k_accuracy_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        binary_top_k_accuracy(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([float("nan"), -1, 0, 3, 1]),
            nan_policy="raise",
        )


###############################################
#     Tests for multiclass_top_k_accuracy     #
###############################################


def test_multiclass_top_k_accuracy() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
        ),
        {"count": 4, "top_2_accuracy": 0.75},
    )


def test_multiclass_top_k_accuracy_k() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            k=[1, 2, 3],
        ),
        {"count": 4, "top_1_accuracy": 0.5, "top_2_accuracy": 0.75, "top_3_accuracy": 1.0},
    )


def test_multiclass_top_k_accuracy_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            k=[1, 2, 3],
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 4,
            "prefix_top_1_accuracy_suffix": 0.5,
            "prefix_top_2_accuracy_suffix": 0.75,
            "prefix_top_3_accuracy_suffix": 1.0,
        },
    )


def test_multiclass_top_k_accuracy_nan_omit() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan"), 0]),
            y_score=np.array(
                [
                    [float("nan"), 0.2, 0.2],
                    [0.3, 0.4, 0.2],
                    [0.2, 0.4, 0.3],
                    [0.7, 0.2, 0.1],
                    [0.5, 0.2, 0.2],
                ]
            ),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_multiclass_top_k_accuracy_nan_omit_y_true() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_multiclass_top_k_accuracy_nan_omit_y_score() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array(
                [[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, float("nan")], [0.1, 0.2, 0.7]]
            ),
            nan_policy="omit",
        ),
        {"count": 3, "top_2_accuracy": 1.0},
    )


def test_multiclass_top_k_accuracy_nan_propagate() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
        ),
        {"count": 4, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_multiclass_top_k_accuracy_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
        ),
        {"count": 4, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_multiclass_top_k_accuracy_nan_propagate_y_score() -> None:
    assert objects_are_equal(
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
        ),
        {"count": 4, "top_2_accuracy": float("nan")},
        equal_nan=True,
    )


def test_multiclass_top_k_accuracy_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
            nan_policy="raise",
        )


def test_multiclass_top_k_accuracy_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, float("nan")]),
            y_score=np.array([[0.5, 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]),
            nan_policy="raise",
        )


def test_multiclass_top_k_accuracy_nan_raise_y_score() -> None:
    with pytest.raises(ValueError, match="'y_score' contains at least one NaN value"):
        multiclass_top_k_accuracy(
            y_true=np.array([0, 1, 2, 2]),
            y_score=np.array(
                [[float("nan"), 0.2, 0.2], [0.3, 0.4, 0.2], [0.2, 0.4, 0.3], [0.7, 0.2, 0.1]]
            ),
            nan_policy="raise",
        )
