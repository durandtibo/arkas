from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.state import PrecisionRecallState

##########################################
#     Tests for PrecisionRecallState     #
##########################################


def test_precision_recall_state_y_true() -> None:
    assert objects_are_equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_precision_recall_state_y_true_2d() -> None:
    assert objects_are_equal(
        PrecisionRecallState(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0]]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true,
        np.array([[1, 0, 0], [1, 1, 1]]),
    )


def test_precision_recall_state_y_pred() -> None:
    assert objects_are_equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_precision_recall_state_y_pred_2d() -> None:
    assert objects_are_equal(
        PrecisionRecallState(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0]]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred,
        np.array([[1, 0, 1], [0, 1, 0]]),
    )


def test_precision_recall_state_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        )


def test_precision_recall_state_incorrect_y_true_ndim() -> None:
    with pytest.raises(ValueError, match="'y_true' must be a 1d or 2d array"):
        PrecisionRecallState(
            y_true=np.ones((2, 3, 4)),
            y_pred=np.ones((2, 3)),
            y_true_name="target",
            y_pred_name="pred",
        )


def test_precision_recall_state_incorrect_y_pred_ndim() -> None:
    with pytest.raises(ValueError, match="'y_pred' must be a 1d or 2d array"):
        PrecisionRecallState(
            y_true=np.ones((2, 3)),
            y_pred=np.ones((2, 3, 4)),
            y_true_name="target",
            y_pred_name="pred",
        )


def test_precision_recall_state_y_true_name() -> None:
    assert (
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_true_name
        == "target"
    )


def test_precision_recall_state_y_pred_name() -> None:
    assert (
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ).y_pred_name
        == "pred"
    )


def test_precision_recall_state_repr() -> None:
    assert repr(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).startswith("PrecisionRecallState(")


def test_precision_recall_state_str() -> None:
    assert str(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).startswith("PrecisionRecallState(")


def test_precision_recall_state_clone() -> None:
    state = PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        label_type="multiclass",
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_precision_recall_state_clone_deep() -> None:
    state = PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        label_type="multiclass",
    )
    cloned_state = state.clone()

    cloned_state.y_true[0] -= 1
    assert state.equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        )
    )
    assert cloned_state.equal(
        PrecisionRecallState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        )
    )


def test_precision_recall_state_clone_shallow() -> None:
    state = PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        label_type="multiclass",
    )
    cloned_state = state.clone(deep=False)

    cloned_state.y_true[0] -= 1
    assert state.equal(
        PrecisionRecallState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        )
    )
    assert cloned_state.equal(
        PrecisionRecallState(
            y_true=np.array([0, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        )
    )


def test_precision_recall_state_equal_true() -> None:
    assert PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_precision_recall_state_equal_false_different_y_true() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_precision_recall_state_equal_false_different_y_pred() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


def test_precision_recall_state_equal_false_different_y_true_name() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="gt",
            y_pred_name="pred",
        )
    )


def test_precision_recall_state_equal_false_different_y_pred_name() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="prediction",
        )
    )


def test_precision_recall_state_equal_false_different_label_type() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        label_type="binary",
    ).equal(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        )
    )


def test_precision_recall_state_equal_false_different_type() -> None:
    assert not PrecisionRecallState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    ).equal(42)
