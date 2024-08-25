from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import fbeta_metrics

###################################
#     Tests for fbeta_metrics     #
###################################


def test_fbeta_metrics() -> None:
    assert objects_are_equal(
        fbeta_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_metrics_binary_correct_1d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
            label_type="binary",
        ),
        {"count": 6, "f1": 1.0},
    )


def test_fbeta_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]), label_type="binary"
        ),
        {"count": 4, "f1": 0.5},
    )


def test_fbeta_metrics_binary_betas() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            betas=[0.5, 1, 2],
            label_type="binary",
        ),
        {"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0},
    )


def test_fbeta_metrics_binary_nans() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            label_type="binary",
        ),
        {"count": 4, "f1": 1.0},
    )


def test_fbeta_metrics_binary_y_true_nan() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            label_type="binary",
        ),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_metrics_binary_y_pred_nan() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
            label_type="binary",
        ),
        {"count": 5, "f1": 1.0},
    )


def test_fbeta_metrics_binary_empty() -> None:
    assert objects_are_equal(
        fbeta_metrics(y_true=np.array([]), y_pred=np.array([]), label_type="binary"),
        {"count": 0, "f1": float("nan")},
        equal_nan=True,
    )


def test_fbeta_metrics_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_f1_suffix": 1.0},
    )


def test_fbeta_metrics_binary_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            label_type="binary",
        )


def test_fbeta_metrics_multiclass_correct_1d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_correct_2d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 0.6666666666666666, 0.0]),
            "count": 6,
            "macro_f1": 0.5555555555555555,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.5555555555555555,
        },
    )


def test_fbeta_metrics_multiclass_betas() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            betas=[0.5, 1, 2],
            label_type="multiclass",
        ),
        {
            "count": 6,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_nans() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 1.0, 1.0]),
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_y_true_nans() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_y_pred_nans() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
            label_type="multiclass",
        ),
        {
            "f1": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multiclass_empty() -> None:
    assert objects_are_allclose(
        fbeta_metrics(y_true=np.array([]), y_pred=np.array([]), label_type="multiclass"),
        {
            "f1": np.array([]),
            "count": 0,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_fbeta_metrics_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_fbeta_metrics_multilabel_1_class_1d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="multilabel",
        ),
        {
            "f1": np.array([1.0]),
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multilabel_1_class_2d() -> None:
    assert objects_are_equal(
        fbeta_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
            label_type="multilabel",
        ),
        {
            "f1": np.array([1.0]),
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
        },
    )


def test_fbeta_metrics_multilabel_3_classes() -> None:
    assert objects_are_allclose(
        fbeta_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            label_type="multilabel",
        ),
        {
            "f1": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_f1": 0.6666666666666666,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.625,
        },
    )


def test_fbeta_metrics_multilabel_betas() -> None:
    assert objects_are_allclose(
        fbeta_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            betas=[0.5, 1, 2],
            label_type="multilabel",
        ),
        {
            "count": 5,
            "f0.5": np.array([1.0, 1.0, 1.0]),
            "macro_f0.5": 1.0,
            "micro_f0.5": 1.0,
            "weighted_f0.5": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "weighted_f1": 1.0,
            "f2": np.array([1.0, 1.0, 1.0]),
            "macro_f2": 1.0,
            "micro_f2": 1.0,
            "weighted_f2": 1.0,
        },
    )


def test_fbeta_metrics_multilabel_empty() -> None:
    assert objects_are_allclose(
        fbeta_metrics(
            y_true=np.array([]),
            y_pred=np.array([]),
            label_type="multilabel",
        ),
        {
            "f1": np.array([]),
            "count": 0,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_fbeta_metrics_multilabel_prefix_suffix() -> None:
    assert objects_are_allclose(
        fbeta_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_fbeta_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect label type: 'incorrect'"):
        fbeta_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )
