from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import binary_recall, multiclass_recall, multilabel_recall, recall

############################
#     Tests for recall     #
############################


def test_recall_auto_binary() -> None:
    assert objects_are_equal(
        recall(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "recall": 1.0},
    )


def test_recall_binary() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "recall": 1.0},
    )


def test_recall_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_recall_suffix": 1.0},
    )


def test_recall_binary_ignore_nan() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            ignore_nan=True,
        ),
        {"count": 5, "recall": 1.0},
    )


def test_recall_auto_multiclass() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_multiclass() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


def test_recall_multiclass_ignore_nan() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            ignore_nan=True,
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_auto_multilabel() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_multilabel() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


def test_recall_multilabel_ignore_nan() -> None:
    assert objects_are_equal(
        recall(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            ignore_nan=True,
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        recall(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


###################################
#     Tests for binary_recall     #
###################################


def test_binary_recall_correct_1d() -> None:
    assert objects_are_equal(
        binary_recall(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_correct_2d() -> None:
    assert objects_are_equal(
        binary_recall(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "recall": 1.0},
    )


def test_binary_recall_incorrect() -> None:
    assert objects_are_equal(
        binary_recall(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0])),
        {"count": 4, "recall": 0.5},
    )


def test_binary_recall_empty() -> None:
    assert objects_are_equal(
        binary_recall(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "recall": float("nan")},
        equal_nan=True,
    )


def test_binary_recall_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_recall(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_recall_suffix": 1.0},
    )


def test_binary_recall_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        binary_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        )


def test_binary_recall_ignore_nan() -> None:
    assert objects_are_equal(
        binary_recall(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            ignore_nan=True,
        ),
        {"count": 4, "recall": 1.0},
    )


def test_binary_recall_ignore_nan_y_true() -> None:
    assert objects_are_equal(
        binary_recall(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            ignore_nan=True,
        ),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_ignore_nan_y_pred() -> None:
    assert objects_are_equal(
        binary_recall(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
            ignore_nan=True,
        ),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_recall(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


#######################################
#     Tests for multiclass_recall     #
#######################################


def test_multiclass_recall_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
        ),
        {
            "count": 6,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.6666666666666666,
            "recall": np.array([1.0, 1.0, 0.0]),
            "weighted_recall": 0.6666666666666666,
        },
        equal_nan=True,
    )


def test_multiclass_recall_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


def test_multiclass_recall_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        )


def test_multiclass_recall_ignore_nan() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            ignore_nan=True,
        ),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_ignore_nan_y_true() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            ignore_nan=True,
        ),
        {
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_ignore_nan_y_pred() -> None:
    assert objects_are_equal(
        multiclass_recall(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
            ignore_nan=True,
        ),
        {
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


#######################################
#     Tests for multilabel_recall     #
#######################################


def test_multilabel_recall_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_recall(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_recall(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "count": 5,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.625,
            "recall": np.array([1.0, 1.0, 0.0]),
            "weighted_recall": 0.625,
        },
    )


def test_multilabel_recall_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_recall(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "recall": np.array([]),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_recall_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_recall(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "recall": np.array([]),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_recall_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


def test_multilabel_recall_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multilabel_recall(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        )


def test_multilabel_recall_ignore_nan() -> None:
    assert objects_are_allclose(
        multilabel_recall(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            ignore_nan=True,
        ),
        {
            "count": 3,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_ignore_nan_y_true() -> None:
    assert objects_are_allclose(
        multilabel_recall(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            ignore_nan=True,
        ),
        {
            "count": 4,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_ignore_nan_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_recall(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            ignore_nan=True,
        ),
        {
            "count": 4,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )
