from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.state import AccuracyState

###################################
#     Tests for AccuracyState     #
###################################


def test_accuracy_state_y_true() -> None:
    assert objects_are_equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_accuracy_state_y_true_2d() -> None:
    assert objects_are_equal(
        AccuracyState(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0]]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_accuracy_state_y_pred() -> None:
    assert objects_are_equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_accuracy_state_y_pred_2d() -> None:
    assert objects_are_equal(
        AccuracyState(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0]]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred,
        np.array([1, 0, 1, 0, 1, 0]),
    )


def test_accuracy_state_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        )


def test_accuracy_state_y_true_name() -> None:
    assert (
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true_name
        == "target"
    )


def test_accuracy_state_y_pred_name() -> None:
    assert (
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred_name
        == "pred"
    )


def test_accuracy_state_nan_policy() -> None:
    assert (
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_accuracy_state_nan_policy_default() -> None:
    assert (
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).nan_policy
        == "propagate"
    )


def test_accuracy_state_nan_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="incorrect",
        )


def test_accuracy_state_repr() -> None:
    assert repr(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).startswith("AccuracyState(")


def test_accuracy_state_str() -> None:
    assert str(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).startswith("AccuracyState(")


def test_accuracy_state_clone() -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        nan_policy="omit",
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_accuracy_state_clone_deep() -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        nan_policy="omit",
    )
    cloned_state = state.clone()

    cloned_state.y_true[0] -= 1
    assert state.equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        )
    )
    assert cloned_state.equal(
        AccuracyState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        )
    )


def test_accuracy_state_clone_shallow() -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        nan_policy="omit",
    )
    cloned_state = state.clone(deep=False)

    cloned_state.y_true[0] -= 1
    assert state.equal(
        AccuracyState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        )
    )
    assert cloned_state.equal(
        AccuracyState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        )
    )


def test_accuracy_state_equal_true() -> None:
    assert AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_accuracy_state_equal_false_different_y_true() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_accuracy_state_equal_false_different_y_pred() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_accuracy_state_equal_false_different_y_true_name() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="gt",
            y_pred_name="pred",
        )
    )


def test_accuracy_state_equal_false_different_y_pred_name() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="prediction",
        )
    )


def test_accuracy_state_equal_false_different_nan_policy() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        )
    )


def test_accuracy_state_equal_false_different_type() -> None:
    assert not AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(42)
