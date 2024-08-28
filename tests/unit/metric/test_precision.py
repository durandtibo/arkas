from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import (
    binary_precision_metrics,
    multiclass_precision_metrics,
    multilabel_precision_metrics,
    precision_metrics,
)
from arkas.metric.precision import find_label_type

#######################################
#     Tests for precision_metrics     #
#######################################


def test_precision_metrics_auto_binary() -> None:
    assert objects_are_equal(
        precision_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "precision": 1.0},
    )


def test_precision_metrics_binary() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "precision": 1.0},
    )


def test_precision_metrics_auto_multiclass() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_metrics_multiclass() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_metrics_auto_multilabel() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "precision": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "weighted_precision": 0.625,
        },
    )


def test_precision_metrics_multilabel() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            label_type="multilabel",
        ),
        {
            "precision": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "weighted_precision": 0.625,
        },
    )


def test_precision_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


##############################################
#     Tests for binary_precision_metrics     #
##############################################


def test_binary_precision_metrics_correct_1d() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_metrics_correct_2d() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "precision": 1.0},
    )


def test_binary_precision_metrics_incorrect() -> None:
    assert objects_are_equal(
        binary_precision_metrics(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0])),
        {"count": 4, "precision": 0.5},
    )


def test_binary_precision_metrics_nans() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 4, "precision": 1.0},
    )


def test_binary_precision_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        ),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
        ),
        {"count": 5, "precision": 1.0},
    )


def test_binary_precision_metrics_empty() -> None:
    assert objects_are_equal(
        binary_precision_metrics(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "precision": float("nan")},
        equal_nan=True,
    )


def test_binary_precision_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_binary_precision_metrics_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


##################################################
#     Tests for multiclass_precision_metrics     #
##################################################


def test_multiclass_precision_metrics_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_metrics_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_metrics_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
        ),
        {
            "precision": np.array([1.0, 0.5, 0.0]),
            "count": 6,
            "macro_precision": 0.5,
            "micro_precision": 0.6666666666666666,
            "weighted_precision": 0.5,
        },
    )


def test_multiclass_precision_metrics_nans() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_metrics_y_true_nans() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_metrics_y_pred_nans() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
        ),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multiclass_precision_metrics_empty() -> None:
    assert objects_are_allclose(
        multiclass_precision_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "precision": np.array([]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_precision_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


##################################################
#     Tests for multilabel_precision_metrics     #
##################################################


def test_multilabel_precision_metrics_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "precision": np.array([1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_metrics_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_precision_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "precision": np.array([1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_multilabel_precision_metrics_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "precision": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "weighted_precision": 0.625,
        },
    )


def test_multilabel_precision_metrics_empty() -> None:
    assert objects_are_allclose(
        multilabel_precision_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "precision": np.array([]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_precision_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


#####################################
#     Tests for find_label_type     #
#####################################


def test_find_label_type_binary() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        )
        == "binary"
    )


def test_find_label_type_binary_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        )
        == "binary"
    )


def test_find_label_type_multiclass() -> None:
    assert (
        find_label_type(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2]))
        == "multiclass"
    )


def test_find_label_type_multiclass_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        )
        == "multiclass"
    )


def test_find_label_type_multiclass_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 1]),
        )
        == "multiclass"
    )


def test_find_label_type_multiclass_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([0, 0, 1, 1, 2, 2, 1]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        )
        == "multiclass"
    )


def test_find_label_type_multilabel() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_nans() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, float("nan")]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_y_true_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 1]]),
        )
        == "multilabel"
    )


def test_find_label_type_multilabel_y_pred_nan() -> None:
    assert (
        find_label_type(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, float("nan"), 1]]),
        )
        == "multilabel"
    )