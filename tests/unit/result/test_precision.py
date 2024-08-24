from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import PrecisionResult
from arkas.result.precision import find_label_type, precision_metrics

#####################################
#     Tests for PrecisionResult     #
#####################################


def test_precision_result_y_true() -> None:
    assert objects_are_equal(
        PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_precision_result_y_true_2d() -> None:
    assert objects_are_equal(
        PrecisionResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([[1, 0, 0], [1, 1, 1]]),
    )


def test_precision_result_y_pred() -> None:
    assert objects_are_equal(
        PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_precision_result_y_pred_2d() -> None:
    assert objects_are_equal(
        PrecisionResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([[0, 1, 0], [1, 0, 1]]),
    )


def test_precision_result_y_true_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_true' must be a 1d or 2d array"):
        PrecisionResult(y_true=np.ones((2, 3, 4)), y_pred=np.ones((2, 3)))


def test_precision_result_y_pred_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_pred' must be a 1d or 2d array"):
        PrecisionResult(y_true=np.ones((2, 3)), y_pred=np.ones((2, 3, 4)))


def test_precision_result_label_type() -> None:
    assert (
        PrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        ).label_type
        == "binary"
    )


def test_precision_result_incorrect_label_type() -> None:
    with pytest.raises(ValueError, match="Incorrect label type: 'incorrect'"):
        PrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


def test_precision_result_repr() -> None:
    assert repr(
        PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("PrecisionResult(")


def test_precision_result_str() -> None:
    assert str(
        PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("PrecisionResult(")


def test_precision_result_equal_true() -> None:
    assert PrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_precision_result_equal_false_different_y_true() -> None:
    assert not PrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(PrecisionResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_precision_result_equal_false_different_y_pred() -> None:
    assert not PrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_precision_result_equal_false_different_type() -> None:
    assert not PrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_precision_result_equal_nan_true() -> None:
    assert PrecisionResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        PrecisionResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_precision_result_compute_metrics_binary_correct() -> None:
    result = PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "precision": 1.0})


def test_precision_result_compute_metrics_binary_incorrect() -> None:
    result = PrecisionResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]))
    assert objects_are_allclose(result.compute_metrics(), {"count": 4, "precision": 0.5})


def test_precision_result_compute_metrics_binary_empty() -> None:
    result = PrecisionResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "precision": float("nan")}, equal_nan=True
    )


def test_precision_result_compute_metrics_binary_prefix_suffix() -> None:
    result = PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_precision_result_compute_metrics_multiclass_correct() -> None:
    result = PrecisionResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_pred=np.array([0, 0, 1, 1, 2, 2]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_result_compute_metrics_multiclass_incorrect() -> None:
    result = PrecisionResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": np.array([1.0, 0.5, 0.0]),
            "count": 6,
            "macro_precision": 0.5,
            "micro_precision": 0.6666666666666666,
            "weighted_precision": 0.5,
        },
    )


def test_precision_result_compute_metrics_multiclass_empty() -> None:
    result = PrecisionResult(y_true=np.array([]), y_pred=np.array([]), label_type="multiclass")
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": np.array([]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_result_compute_metrics_multiclass_prefix_suffix() -> None:
    result = PrecisionResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_precision_suffix": 1.0,
            "prefix_micro_precision_suffix": 1.0,
            "prefix_weighted_precision_suffix": 1.0,
        },
    )


def test_precision_result_compute_metrics_multilabel_1_class() -> None:
    result = PrecisionResult(
        y_true=np.array([[1], [0], [0], [1], [1]]), y_pred=np.array([[1], [0], [0], [1], [1]])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": np.array([1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_result_compute_metrics_multilabel_3_classes() -> None:
    result = PrecisionResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "precision": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "weighted_precision": 0.625,
        },
    )


def test_precision_result_compute_metrics_multilabel_empty() -> None:
    result = PrecisionResult(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3)))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "precision": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_result_compute_metrics_multilabel_prefix_suffix() -> None:
    result = PrecisionResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_precision_suffix": np.array([1.0, 1.0, 0.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_precision_suffix": 0.6666666666666666,
            "prefix_micro_precision_suffix": 0.7142857142857143,
            "prefix_weighted_precision_suffix": 0.625,
        },
    )


def test_precision_result_generate_figures() -> None:
    result = PrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_precision_result_generate_figures_empty() -> None:
    result = PrecisionResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


#######################################
#     Tests for precision_metrics     #
#######################################


def test_precision_metrics() -> None:
    assert objects_are_equal(
        precision_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"count": 5, "precision": 1.0},
    )


def test_precision_metrics_binary_correct_1d() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), label_type="binary"
        ),
        {"count": 5, "precision": 1.0},
    )


def test_precision_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [1, 1, 1]]),
            label_type="binary",
        ),
        {"count": 6, "precision": 1.0},
    )


def test_precision_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]), label_type="binary"
        ),
        {"count": 4, "precision": 0.5},
    )


def test_precision_metrics_binary_empty() -> None:
    assert objects_are_equal(
        precision_metrics(y_true=np.array([]), y_pred=np.array([]), label_type="binary"),
        {"count": 0, "precision": float("nan")},
        equal_nan=True,
    )


def test_precision_metrics_binary_prefix_suffix() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="binary",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_precision_metrics_binary_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes:"):
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            label_type="binary",
        )


def test_precision_metrics_multiclass_correct_1d() -> None:
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


def test_precision_metrics_multiclass_correct_2d() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]),
            y_pred=np.array([[0, 0, 1], [1, 2, 2]]),
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


def test_precision_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
            label_type="multiclass",
        ),
        {
            "precision": np.array([1.0, 0.5, 0.0]),
            "count": 6,
            "macro_precision": 0.5,
            "micro_precision": 0.6666666666666666,
            "weighted_precision": 0.5,
        },
    )


def test_precision_metrics_multiclass_empty() -> None:
    assert objects_are_allclose(
        precision_metrics(y_true=np.array([]), y_pred=np.array([]), label_type="multiclass"),
        {
            "precision": np.array([]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_metrics_multiclass_prefix_suffix() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            label_type="multiclass",
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


def test_precision_metrics_multilabel_1_class_1d() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="multilabel",
        ),
        {
            "precision": np.array([1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_metrics_multilabel_1_class_2d() -> None:
    assert objects_are_equal(
        precision_metrics(
            y_true=np.array([[1], [0], [0], [1], [1]]),
            y_pred=np.array([[1], [0], [0], [1], [1]]),
            label_type="multilabel",
        ),
        {
            "precision": np.array([1.0]),
            "count": 5,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "weighted_precision": 1.0,
        },
    )


def test_precision_metrics_multilabel_3_classes() -> None:
    assert objects_are_allclose(
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


def test_precision_metrics_multilabel_empty() -> None:
    assert objects_are_allclose(
        precision_metrics(
            y_true=np.array([]),
            y_pred=np.array([]),
            label_type="multilabel",
        ),
        {
            "precision": np.array([]),
            "count": 0,
            "macro_precision": float("nan"),
            "micro_precision": float("nan"),
            "weighted_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_precision_metrics_multilabel_prefix_suffix() -> None:
    assert objects_are_allclose(
        precision_metrics(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            label_type="multilabel",
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_precision_suffix": np.array([1.0, 1.0, 0.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_precision_suffix": 0.6666666666666666,
            "prefix_micro_precision_suffix": 0.7142857142857143,
            "prefix_weighted_precision_suffix": 0.625,
        },
    )


def test_precision_metrics_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect label type: 'incorrect'"):
        precision_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
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


def test_find_label_type_multiclass() -> None:
    assert (
        find_label_type(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2]))
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
