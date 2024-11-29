from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import (
    binary_confusion_matrix,
    confusion_matrix,
    multiclass_confusion_matrix,
    multilabel_confusion_matrix,
)

######################################
#     Tests for confusion_matrix     #
######################################


def test_confusion_matrix_binary_auto_binary() -> None:
    assert objects_are_equal(
        confusion_matrix(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_confusion_matrix_binary() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_confusion_matrix_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array([[2, 0], [0, 3]]),
            "prefix_count_suffix": 5,
            "prefix_false_negative_rate_suffix": 0.0,
            "prefix_false_negative_suffix": 0,
            "prefix_false_positive_rate_suffix": 0.0,
            "prefix_false_positive_suffix": 0,
            "prefix_true_negative_rate_suffix": 1.0,
            "prefix_true_negative_suffix": 2,
            "prefix_true_positive_rate_suffix": 1.0,
            "prefix_true_positive_suffix": 3,
        },
    )


def test_confusion_matrix_binary_drop_nan() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_confusion_matrix_multiclass() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 1, 1, 2, 2, 2]),
            label_type="multiclass",
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_confusion_matrix_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 1, 1, 2, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "prefix_count_suffix": 6,
        },
    )


def test_confusion_matrix_multiclass_drop_nan() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            label_type="multiclass",
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_confusion_matrix_auto_multilabel() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_confusion_matrix_multilabel() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_confusion_matrix_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array(
                [[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]
            ),
            "prefix_count_suffix": 5,
        },
    )


def test_confusion_matrix_multilabel_drop_nan() -> None:
    assert objects_are_equal(
        confusion_matrix(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            label_type="multilabel",
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_confusion_matrix_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'label_type': incorrect"):
        confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


#############################################
#     Tests for binary_confusion_matrix     #
#############################################


def test_binary_confusion_matrix_correct_1d() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_binary_confusion_matrix_correct_2d() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 0], [1, 1, 1]])
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 4]]),
            "count": 6,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 4,
        },
    )


def test_binary_confusion_matrix_incorrect() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 1, 0, 0])),
        {
            "confusion_matrix": np.array([[0, 2], [3, 0]]),
            "count": 5,
            "false_negative_rate": 1.0,
            "false_negative": 3,
            "false_positive_rate": 1.0,
            "false_positive": 2,
            "true_negative_rate": 0.0,
            "true_negative": 0,
            "true_positive_rate": 0.0,
            "true_positive": 0,
        },
    )


def test_binary_confusion_matrix_empty() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(y_true=np.array([]), y_pred=np.array([])),
        {
            "confusion_matrix": np.array([[0, 0], [0, 0]]),
            "count": 0,
            "false_negative_rate": float("nan"),
            "false_negative": 0,
            "false_positive_rate": float("nan"),
            "false_positive": 0,
            "true_negative_rate": float("nan"),
            "true_negative": 0,
            "true_positive_rate": float("nan"),
            "true_positive": 0,
        },
        equal_nan=True,
    )


def test_binary_confusion_matrix_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array([[2, 0], [0, 3]]),
            "prefix_count_suffix": 5,
            "prefix_false_negative_rate_suffix": 0.0,
            "prefix_false_negative_suffix": 0,
            "prefix_false_positive_rate_suffix": 0.0,
            "prefix_false_positive_suffix": 0,
            "prefix_true_negative_rate_suffix": 1.0,
            "prefix_true_negative_suffix": 2,
            "prefix_true_positive_rate_suffix": 1.0,
            "prefix_true_positive_suffix": 3,
        },
    )


def test_binary_confusion_matrix_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        binary_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        )


def test_binary_confusion_matrix_drop_nan() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_binary_confusion_matrix_drop_nan_y_true() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


def test_binary_confusion_matrix_drop_nan_y_pred() -> None:
    assert objects_are_equal(
        binary_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
        },
    )


#################################################
#     Tests for multiclass_confusion_matrix     #
#################################################


def test_multiclass_confusion_matrix_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([[0, 1, 1], [2, 2, 2]]), y_pred=np.array([[0, 1, 1], [2, 2, 2]])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_incorrect() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 2, 0, 1, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 1], [0, 1, 1], [0, 1, 1]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_empty() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(y_true=np.array([]), y_pred=np.array([])),
        {
            "confusion_matrix": np.zeros((0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multiclass_confusion_matrix_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 1, 1, 2, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "prefix_count_suffix": 6,
        },
    )


def test_multiclass_confusion_matrix_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        )


def test_multiclass_confusion_matrix_drop_nan() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_drop_nan_y_true() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, 1]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_drop_nan_y_pred() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix(
            y_true=np.array([0, 1, 1, 2, 2, 2, 1]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


#################################################
#     Tests for multilabel_confusion_matrix     #
#################################################


def test_multilabel_confusion_matrix_correct() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
        show_difference=True,
    )


def test_multilabel_confusion_matrix_incorrect() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[0, 2], [3, 0]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_confusion_matrix(y_true=np.array([]), y_pred=np.array([])),
        {
            "confusion_matrix": np.zeros((0, 0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multilabel_confusion_matrix_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_confusion_matrix(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "confusion_matrix": np.zeros((0, 0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multilabel_confusion_matrix_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_confusion_matrix_suffix": np.array(
                [[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]
            ),
            "prefix_count_suffix": 5,
        },
    )


def test_multilabel_confusion_matrix_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        )


def test_multilabel_confusion_matrix_drop_nan() -> None:
    assert objects_are_allclose(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 1]], [[1, 0], [0, 2]], [[2, 0], [0, 1]]]),
            "count": 3,
        },
    )


def test_multilabel_confusion_matrix_drop_nan_y_true() -> None:
    assert objects_are_allclose(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]),
            "count": 4,
        },
    )


def test_multilabel_confusion_matrix_drop_nan_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_confusion_matrix(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            drop_nan=True,
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 2]], [[2, 0], [0, 2]], [[2, 0], [0, 2]]]),
            "count": 4,
        },
    )
