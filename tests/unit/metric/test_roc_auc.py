from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import roc_auc_metrics

#####################################
#     Tests for roc_auc_metrics     #
#####################################


def test_roc_auc_metrics() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_binary_correct() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]),
            y_score=np.array([[2, -1, 0], [3, 1, -2]]),
            label_type="binary",
        ),
        {"count": 6, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1]),
            y_score=np.array([-1, 1, 0, -2]),
            label_type="binary",
        ),
        {"count": 4, "roc_auc": 0.0},
    )


def test_roc_auc_metrics_binary_empty() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([]),
            y_score=np.array([]),
            label_type="binary",
        ),
        {"count": 0, "roc_auc": float("nan")},
        equal_nan=True,
    )


def test_roc_auc_metrics_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_roc_auc_suffix": 1.0},
    )


def test_roc_auc_metrics_binary_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes:"):
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1, 6]),
            label_type="binary",
        )


def test_roc_auc_metrics_binary_nans() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1, float("nan")]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_binary_y_true_nan() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1, 0]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_binary_y_score_nan() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_score=np.array([2, -1, 0, float("nan"), 1, -3]),
            label_type="binary",
        ),
        {"count": 5, "roc_auc": 1.0},
    )


def test_roc_auc_metrics_multiclass_correct() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.3, 0.5],
                    [0.4, 0.4, 0.2],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "macro_roc_auc": 0.8333333333333334,
            "micro_roc_auc": 0.826388888888889,
            "roc_auc": np.array([0.9375, 0.8125, 0.75]),
            "weighted_roc_auc": 0.8333333333333334,
        },
    )


def test_roc_auc_metrics_multiclass_empty_1d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([]),
            y_score=np.array([]),
            label_type="multiclass",
        ),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_auc_metrics_multiclass_empty_2d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.ones((0,)),
            y_score=np.ones((0, 3)),
            label_type="multiclass",
        ),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([float("nan"), float("nan"), float("nan")]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_auc_metrics_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_roc_auc_metrics_multiclass_nans() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 4,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multiclass_y_true_nan() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.5, 0.3, 0.2],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multiclass_y_score_nan() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, 0]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, float("nan")],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [float("nan"), float("nan"), float("nan")],
                    [0.7, 0.2, 0.1],
                ]
            ),
            label_type="multiclass",
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multilabel_1_class_1d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multilabel_1_class_2d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_score=np.array([[2], [-1], [0], [3], [1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "roc_auc": np.array([1.0]),
            "weighted_roc_auc": 1.0,
        },
    )


def test_roc_auc_metrics_multilabel_3_classes() -> None:
    assert objects_are_allclose(
        roc_auc_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "macro_roc_auc": 0.6666666666666666,
            "micro_roc_auc": 0.5446428571428571,
            "roc_auc": np.array([1.0, 1.0, 0.0]),
            "weighted_roc_auc": 0.625,
        },
    )


def test_roc_auc_metrics_multilabel_empty_1d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(y_true=np.array([]), y_score=np.array([]), label_type="multilabel"),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_auc_metrics_multilabel_empty_2d() -> None:
    assert objects_are_equal(
        roc_auc_metrics(y_true=np.ones((0, 3)), y_score=np.ones((0, 3)), label_type="multilabel"),
        {
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "roc_auc": np.array([float("nan"), float("nan"), float("nan")]),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_roc_auc_metrics_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        roc_auc_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_roc_auc_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect label type: 'incorrect'"):
        roc_auc_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            label_type="incorrect",
        )
