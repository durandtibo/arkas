from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import (
    binary_jaccard_metrics,
    jaccard_metrics,
    multiclass_jaccard_metrics,
    multilabel_jaccard_metrics,
)

#####################################
#     Tests for jaccard_metrics     #
#####################################


def test_jaccard_metrics_auto_binary() -> None:
    assert objects_are_equal(
        jaccard_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "jaccard": 1.0},
    )


def test_jaccard_metrics_binary() -> None:
    assert objects_are_equal(
        jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_jaccard_metrics_auto_multiclass() -> None:
    assert objects_are_equal(
        jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_metrics_multiclass() -> None:
    assert objects_are_equal(
        jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_metrics_auto_multilabel() -> None:
    assert objects_are_equal(
        jaccard_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 5,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_metrics_multilabel() -> None:
    assert objects_are_equal(
        jaccard_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


############################################
#     Tests for binary_jaccard_metrics     #
############################################


def test_binary_jaccard_metrics_correct_1d() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_metrics_correct_2d() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "jaccard": 1.0},
    )


def test_binary_jaccard_metrics_incorrect() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1, 1]),
        ),
        {"count": 6, "jaccard": 0.6},
    )


def test_binary_jaccard_metrics_nans() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
        ),
        {"count": 4, "jaccard": 1.0},
    )


def test_binary_jaccard_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_metrics_empty() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "jaccard": float("nan")},
        equal_nan=True,
    )


def test_binary_jaccard_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_jaccard_suffix": 1.0},
    )


def test_binary_jaccard_metrics_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


###############################################
#     Tests for multiclass_jaccard_metrics     #
###############################################


def test_multiclass_jaccard_metrics_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_metrics_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_metrics_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 0.5, 0.0]),
            "macro_jaccard": 0.5,
            "micro_jaccard": 0.5,
            "weighted_jaccard": 0.5,
        },
    )


def test_multiclass_jaccard_metrics_nans() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_metrics_y_true_nans() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_metrics_y_pred_nans() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_metrics_empty() -> None:
    assert objects_are_allclose(
        multiclass_jaccard_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "jaccard": np.array([]),
            "count": 0,
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_jaccard_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        multiclass_jaccard_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )


################################################
#     Tests for multilabel_jaccard_metrics     #
################################################


def test_multilabel_jaccard_metrics_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_jaccard_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "jaccard": np.array([1.0]),
            "count": 5,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_metrics_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_jaccard_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "jaccard": np.array([1.0]),
            "count": 5,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_metrics_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "jaccard": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_jaccard": 0.6666666666666666,
            "micro_jaccard": 0.5,
            "weighted_jaccard": 0.625,
        },
    )


def test_multilabel_jaccard_metrics_nans() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        ),
        {
            "count": 3,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_metrics_y_true_nans() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 4,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_metrics_y_pred_nans() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 4,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_metrics_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "jaccard": np.array([]),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_jaccard_metrics_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "count": 0,
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "jaccard": np.array([]),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_jaccard_metrics_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_jaccard_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )
