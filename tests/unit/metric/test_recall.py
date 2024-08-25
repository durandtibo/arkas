from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import (
    binary_recall_metrics,
    multiclass_recall_metrics,
    multilabel_recall_metrics,
    recall_metrics,
)

####################################
#     Tests for recall_metrics     #
####################################


def test_recall_metrics_auto_binary() -> None:
    assert objects_are_equal(
        recall_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "recall": 1.0},
    )


def test_recall_metrics_binary() -> None:
    assert objects_are_equal(
        recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "recall": 1.0},
    )


def test_recall_metrics_auto_multiclass() -> None:
    assert objects_are_equal(
        recall_metrics(
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


def test_recall_metrics_multiclass() -> None:
    assert objects_are_equal(
        recall_metrics(
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


def test_recall_metrics_auto_multilabel() -> None:
    assert objects_are_equal(
        recall_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_recall_metrics_multilabel() -> None:
    assert objects_are_equal(
        recall_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "recall": np.array([1.0, 1.0, 1.0]),
            "weighted_recall": 1.0,
        },
    )


def test_recall_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


###########################################
#     Tests for binary_recall_metrics     #
###########################################


def test_binary_recall_metrics_correct_1d() -> None:
    assert objects_are_equal(
        binary_recall_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_metrics_correct_2d() -> None:
    assert objects_are_equal(
        binary_recall_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "recall": 1.0},
    )


def test_binary_recall_metrics_incorrect() -> None:
    assert objects_are_equal(
        binary_recall_metrics(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0])),
        {"count": 4, "recall": 0.5},
    )


def test_binary_recall_metrics_nans() -> None:
    assert objects_are_equal(
        binary_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 4, "recall": 1.0},
    )


def test_binary_recall_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        binary_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        ),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        binary_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
        ),
        {"count": 5, "recall": 1.0},
    )


def test_binary_recall_metrics_empty() -> None:
    assert objects_are_equal(
        binary_recall_metrics(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "recall": float("nan")},
        equal_nan=True,
    )


def test_binary_recall_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_recall_suffix": 1.0},
    )


def test_binary_recall_metrics_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


###############################################
#     Tests for multiclass_recall_metrics     #
###############################################


def test_multiclass_recall_metrics_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
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


def test_multiclass_recall_metrics_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_metrics_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_recall_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
        ),
        {
            "recall": np.array([1.0, 1.0, 0.0]),
            "count": 6,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.6666666666666666,
            "weighted_recall": 0.6666666666666666,
        },
    )


def test_multiclass_recall_metrics_nans() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_metrics_y_true_nans() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_metrics_y_pred_nans() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
        ),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multiclass_recall_metrics_empty() -> None:
    assert objects_are_allclose(
        multiclass_recall_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "recall": np.array([]),
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_recall_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_recall_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


###############################################
#     Tests for multilabel_recall_metrics     #
###############################################


def test_multilabel_recall_metrics_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_recall_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "recall": np.array([1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_metrics_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_recall_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "recall": np.array([1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_multilabel_recall_metrics_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_recall_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "recall": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.625,
            "weighted_recall": 0.625,
        },
    )


def test_multilabel_recall_metrics_empty() -> None:
    assert objects_are_allclose(
        multilabel_recall_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "recall": np.array([]),
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_recall_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_recall_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_weighted_recall_suffix": 1.0,
        },
    )
