from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric.ap import average_precision_metrics, find_label_type

###############################################
#     Tests for average_precision_metrics     #
###############################################


def test_average_precision_metrics() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ),
        {"count": 5, "average_precision": 1.0},
    )


def test_average_precision_metrics_binary_correct() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
        ),
        {"count": 5, "average_precision": 1.0},
    )


def test_average_precision_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]),
            y_score=np.array([[2, -1, 0], [3, 1, -2]]),
            label_type="binary",
        ),
        {"count": 6, "average_precision": 1.0},
    )


def test_average_precision_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1]),
            y_score=np.array([-1, 1, 0, -2]),
            label_type="binary",
        ),
        {"count": 4, "average_precision": 0.41666666666666663},
    )


def test_average_precision_metrics_binary_empty() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([]),
            y_score=np.array([]),
            label_type="binary",
        ),
        {"count": 0, "average_precision": float("nan")},
        equal_nan=True,
    )


def test_average_precision_metrics_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_average_precision_suffix": 1.0},
    )


def test_average_precision_metrics_binary_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes:"):
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1, 6]),
            label_type="binary",
        )


def test_average_precision_metrics_binary_nans() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1, float("nan")]),
            label_type="binary",
        ),
        {"count": 5, "average_precision": 1.0},
    )


def test_average_precision_metrics_binary_y_true_nan() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_score=np.array([2, -1, 0, 3, 1, 0]),
            label_type="binary",
        ),
        {"count": 5, "average_precision": 1.0},
    )


def test_average_precision_metrics_binary_y_score_nan() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_score=np.array([2, -1, 0, float("nan"), 1, -3]),
            label_type="binary",
        ),
        {"count": 5, "average_precision": 1.0},
    )


def test_average_precision_metrics_multiclass_correct() -> None:
    assert objects_are_equal(
        average_precision_metrics(
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
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        average_precision_metrics(
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
            "average_precision": np.array([0.8333333333333333, 0.75, 0.75]),
            "count": 6,
            "macro_average_precision": 0.7777777777777777,
            "micro_average_precision": 0.75,
            "weighted_average_precision": 0.7777777777777777,
        },
    )


def test_average_precision_metrics_multiclass_empty_1d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([]),
            y_score=np.array([]),
            label_type="multiclass",
        ),
        {
            "average_precision": np.array([]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_metrics_multiclass_empty_2d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.ones((0,)),
            y_score=np.ones((0, 3)),
            label_type="multiclass",
        ),
        {
            "average_precision": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_metrics_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        average_precision_metrics(
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
            "prefix_average_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_average_precision_suffix": 1.0,
            "prefix_micro_average_precision_suffix": 1.0,
            "prefix_weighted_average_precision_suffix": 1.0,
        },
    )


def test_average_precision_metrics_multiclass_nans() -> None:
    assert objects_are_equal(
        average_precision_metrics(
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
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 4,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multiclass_y_true_nan() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.2],
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
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multiclass_y_score_nan() -> None:
    assert objects_are_equal(
        average_precision_metrics(
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
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multilabel_1_class_1d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="multilabel",
        ),
        {
            "average_precision": np.array([1.0]),
            "count": 5,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multilabel_1_class_2d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_score=np.array([[2], [-1], [0], [3], [1]]),
            label_type="multilabel",
        ),
        {
            "average_precision": np.array([1.0]),
            "count": 5,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_metrics_multilabel_3_classes() -> None:
    assert objects_are_allclose(
        average_precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
            label_type="multilabel",
        ),
        {
            "average_precision": np.array([1.0, 1.0, 0.4777777777777778]),
            "count": 5,
            "macro_average_precision": 0.825925925925926,
            "micro_average_precision": 0.5884199134199134,
            "weighted_average_precision": 0.8041666666666667,
        },
    )


def test_average_precision_metrics_multilabel_empty_1d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([]), y_score=np.array([]), label_type="multilabel"
        ),
        {
            "average_precision": np.array([]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_metrics_multilabel_empty_2d() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.ones((0, 3)), y_score=np.ones((0, 3)), label_type="multilabel"
        ),
        {
            "average_precision": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_metrics_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        average_precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_average_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_average_precision_suffix": 1.0,
            "prefix_micro_average_precision_suffix": 1.0,
            "prefix_weighted_average_precision_suffix": 1.0,
        },
    )


def test_average_precision_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect label type: 'incorrect'"):
        average_precision_metrics(
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


#####################################
#     Tests for find_label_type     #
#####################################


def test_find_label_type_binary() -> None:
    assert find_label_type(y_true=np.ones(5), y_score=np.ones(5)) == "binary"


def test_find_label_type_multiclass() -> None:
    assert find_label_type(y_true=np.ones(5), y_score=np.ones((5, 3))) == "multiclass"


def test_find_label_type_multilabel() -> None:
    assert find_label_type(y_true=np.ones((5, 3)), y_score=np.ones((5, 3))) == "multilabel"


def test_find_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Could not find the label type"):
        find_label_type(y_true=np.ones(5), y_score=np.ones((5, 2, 3)))
