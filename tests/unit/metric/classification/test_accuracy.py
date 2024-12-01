from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import accuracy, balanced_accuracy

##############################
#     Tests for accuracy     #
##############################


def test_accuracy_binary_correct() -> None:
    assert objects_are_equal(
        accuracy(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_binary_correct_2d() -> None:
    assert objects_are_equal(
        accuracy(y_true=np.array([[1, 0, 0], [1, 1, 0]]), y_pred=np.array([[1, 0, 0], [1, 1, 0]])),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_binary_incorrect() -> None:
    assert objects_are_equal(
        accuracy(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0])),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_multiclass_correct() -> None:
    assert objects_are_equal(
        accuracy(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        accuracy(y_true=np.array([0, 0, 1, 1, 2]), y_pred=np.array([0, 0, 1, 1, 1])),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


def test_accuracy_empty() -> None:
    assert objects_are_equal(
        accuracy(y_true=np.array([]), y_pred=np.array([])),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_prefix_suffix() -> None:
    assert objects_are_equal(
        accuracy(
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


def test_accuracy_nan_omit() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="omit",
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_nan_omit_y_true() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="omit",
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="omit",
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_nan_propagate() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="propagate",
        ),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="propagate",
        ),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="propagate",
        ),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="raise",
        )


def test_accuracy_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            nan_policy="raise",
        )


def test_accuracy_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        accuracy(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            nan_policy="raise",
        )


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
