from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric import (
    binary_confusion_matrix_metrics,
    confusion_matrix_metrics,
    multiclass_confusion_matrix_metrics,
    multilabel_confusion_matrix_metrics,
)

##############################################
#     Tests for confusion_matrix_metrics     #
##############################################


def test_confusion_matrix_metrics_binary_auto_binary() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
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


def test_confusion_matrix_metrics_binary() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
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


def test_confusion_matrix_metrics_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
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


def test_confusion_matrix_metrics_multiclass() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
            y_true=np.array([0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 1, 1, 2, 2, 2]),
            label_type="multiclass",
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_confusion_matrix_metrics_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
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


def test_confusion_matrix_metrics_auto_multilabel() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_confusion_matrix_metrics_multilabel() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_confusion_matrix_metrics_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        confusion_matrix_metrics(
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


def test_confusion_matrix_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


#####################################################
#     Tests for binary_confusion_matrix_metrics     #
#####################################################


def test_binary_confusion_matrix_metrics_correct_1d() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
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


def test_binary_confusion_matrix_metrics_correct_2d() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
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


def test_binary_confusion_matrix_metrics_incorrect() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 1, 0, 0])
        ),
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


def test_binary_confusion_matrix_metrics_nans() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
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


def test_binary_confusion_matrix_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
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


def test_binary_confusion_matrix_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
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


def test_binary_confusion_matrix_metrics_empty() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(y_true=np.array([]), y_pred=np.array([])),
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


def test_binary_confusion_matrix_metrics_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_confusion_matrix_metrics(
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


#########################################################
#     Tests for multiclass_confusion_matrix_metrics     #
#########################################################


def test_multiclass_confusion_matrix_metrics_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([0, 1, 1, 2, 2, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([[0, 1, 1], [2, 2, 2]]), y_pred=np.array([[0, 1, 1], [2, 2, 2]])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_incorrect() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([0, 1, 2, 0, 1, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
        ),
        {
            "confusion_matrix": np.array([[1, 0, 1], [0, 1, 1], [0, 1, 1]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_nans() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, 1]),
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
            y_true=np.array([0, 1, 1, 2, 2, 2, 1]),
            y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        ),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_metrics_empty() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "confusion_matrix": np.zeros((0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multiclass_confusion_matrix_metrics_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_confusion_matrix_metrics(
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


#########################################################
#     Tests for multilabel_confusion_matrix_metrics     #
#########################################################


def test_multilabel_confusion_matrix_metrics_correct() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_metrics_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_metrics_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
        show_difference=True,
    )


def test_multilabel_confusion_matrix_metrics_incorrect() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[0, 2], [3, 0]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_metrics_empty() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "confusion_matrix": np.zeros((0, 0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multilabel_confusion_matrix_metrics_correct_prefix_suffix() -> None:
    assert objects_are_equal(
        multilabel_confusion_matrix_metrics(
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
