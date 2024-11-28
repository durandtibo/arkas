from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.metric import binary_jaccard, jaccard, multiclass_jaccard, multilabel_jaccard

#############################
#     Tests for jaccard     #
#############################


def test_jaccard_auto_binary() -> None:
    assert objects_are_equal(
        jaccard(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "jaccard": 1.0},
    )


def test_jaccard_binary() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_jaccard_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_jaccard_suffix": 1.0},
    )


def test_jaccard_binary_drop_nan() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            label_type="binary",
            drop_nan=True,
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_jaccard_auto_multiclass() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_multiclass() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 6,
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )


def test_jaccard_multiclass_drop_nan() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            label_type="multiclass",
            drop_nan=True,
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_auto_multilabel() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_multilabel() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_multilabel_prefix_suffix() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )


def test_jaccard_multilabel_drop_nan() -> None:
    assert objects_are_equal(
        jaccard(
            y_true=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
            ),
            y_pred=np.array(
                [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
            ),
            label_type="multilabel",
            drop_nan=True,
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_jaccard_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        jaccard(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


####################################
#     Tests for binary_jaccard     #
####################################


def test_binary_jaccard_correct_1d() -> None:
    assert objects_are_equal(
        binary_jaccard(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_correct_2d() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
        ),
        {"count": 6, "jaccard": 1.0},
    )


def test_binary_jaccard_incorrect() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1, 1]),
        ),
        {"count": 6, "jaccard": 0.6},
    )


def test_binary_jaccard_empty() -> None:
    assert objects_are_equal(
        binary_jaccard(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "jaccard": float("nan")},
        equal_nan=True,
    )


def test_binary_jaccard_prefix_suffix() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_jaccard_suffix": 1.0},
    )


def test_binary_jaccard_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        binary_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        )


def test_binary_jaccard_drop_nan() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 1]),
            drop_nan=True,
        ),
        {"count": 4, "jaccard": 1.0},
    )


def test_binary_jaccard_drop_nan_y_true() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 1]),
            drop_nan=True,
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_drop_nan_y_pred() -> None:
    assert objects_are_equal(
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, float("nan"), 0]),
            drop_nan=True,
        ),
        {"count": 5, "jaccard": 1.0},
    )


def test_binary_jaccard_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        binary_jaccard(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


########################################
#     Tests for multiclass_jaccard     #
########################################


def test_multiclass_jaccard_correct_1d() -> None:
    assert objects_are_equal(
        multiclass_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_correct_2d() -> None:
    assert objects_are_equal(
        multiclass_jaccard(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_incorrect() -> None:
    assert objects_are_allclose(
        multiclass_jaccard(
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


def test_multiclass_jaccard_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multiclass_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
        )


def test_multiclass_jaccard_drop_nan() -> None:
    assert objects_are_equal(
        multiclass_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, float("nan"), 2]),
            drop_nan=True,
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_drop_nan_y_true() -> None:
    assert objects_are_equal(
        multiclass_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, 2]),
            drop_nan=True,
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_drop_nan_y_pred() -> None:
    assert objects_are_equal(
        multiclass_jaccard(
            y_true=np.array([0, 0, 1, 1, 2, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, float("nan"), 2, 2]),
            drop_nan=True,
        ),
        {
            "count": 6,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


########################################
#     Tests for multilabel_jaccard     #
########################################


def test_multilabel_jaccard_1_class_1d() -> None:
    assert objects_are_equal(
        multilabel_jaccard(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_1_class_2d() -> None:
    assert objects_are_equal(
        multilabel_jaccard(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_3_classes() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ),
        {
            "count": 5,
            "jaccard": np.array([1.0, 1.0, 0.0]),
            "macro_jaccard": 0.6666666666666666,
            "micro_jaccard": 0.5,
            "weighted_jaccard": 0.625,
        },
    )


def test_multilabel_jaccard_empty_1d() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(y_true=np.array([]), y_pred=np.array([])),
        {
            "count": 0,
            "jaccard": np.array([]),
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_jaccard_empty_2d() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3))),
        {
            "count": 0,
            "jaccard": np.array([]),
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_jaccard_prefix_suffix() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )


def test_multilabel_jaccard_nan() -> None:
    with pytest.raises(ValueError, match="Input.* contains NaN"):
        multilabel_jaccard(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
        )


def test_multilabel_jaccard_drop_nan() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [float("nan"), 0, 1]]),
            drop_nan=True,
        ),
        {
            "count": 3,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_drop_nan_y_true() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(
            y_true=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            drop_nan=True,
        ),
        {
            "count": 4,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )


def test_multilabel_jaccard_drop_nan_y_pred() -> None:
    assert objects_are_allclose(
        multilabel_jaccard(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, float("nan")], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            drop_nan=True,
        ),
        {
            "count": 4,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "weighted_jaccard": 1.0,
        },
    )
