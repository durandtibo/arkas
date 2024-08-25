from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import RecallResult

##################################
#     Tests for RecallResult     #
##################################


def test_recall_result_y_true() -> None:
    assert objects_are_equal(
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_recall_result_y_true_2d() -> None:
    assert objects_are_equal(
        RecallResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([[1, 0, 0], [1, 1, 1]]),
    )


def test_recall_result_y_pred() -> None:
    assert objects_are_equal(
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_recall_result_y_pred_2d() -> None:
    assert objects_are_equal(
        RecallResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([[0, 1, 0], [1, 0, 1]]),
    )


def test_recall_result_y_true_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_true' must be a 1d or 2d array"):
        RecallResult(y_true=np.ones((2, 3, 4)), y_pred=np.ones((2, 3)))


def test_recall_result_y_pred_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_pred' must be a 1d or 2d array"):
        RecallResult(y_true=np.ones((2, 3)), y_pred=np.ones((2, 3, 4)))


def test_recall_result_label_type() -> None:
    assert (
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])).label_type
        == "binary"
    )


def test_recall_result_incorrect_label_type() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        RecallResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


def test_recall_result_repr() -> None:
    assert repr(
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("RecallResult(")


def test_recall_result_str() -> None:
    assert str(
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("RecallResult(")


def test_recall_result_equal_true() -> None:
    assert RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).equal(
        RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )


def test_recall_result_equal_false_different_y_true() -> None:
    assert not RecallResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(RecallResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_recall_result_equal_false_different_y_pred() -> None:
    assert not RecallResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_recall_result_equal_false_different_label_type() -> None:
    assert not RecallResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        RecallResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 0]),
            label_type="multiclass",
        )
    )


def test_recall_result_equal_false_different_type() -> None:
    assert not RecallResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_recall_result_equal_nan_true() -> None:
    assert RecallResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        RecallResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_recall_result_compute_metrics_binary_correct() -> None:
    result = RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "recall": 1.0})


def test_recall_result_compute_metrics_binary_incorrect() -> None:
    result = RecallResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]))
    assert objects_are_allclose(result.compute_metrics(), {"count": 4, "recall": 0.5})


def test_recall_result_compute_metrics_binary_empty() -> None:
    result = RecallResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "recall": float("nan")}, equal_nan=True
    )


def test_recall_result_compute_metrics_binary_prefix_suffix() -> None:
    result = RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_recall_suffix": 1.0},
    )


def test_recall_result_compute_metrics_multiclass_correct() -> None:
    result = RecallResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_pred=np.array([0, 0, 1, 1, 2, 2]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "recall": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_result_compute_metrics_multiclass_incorrect() -> None:
    result = RecallResult(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "recall": np.array([1.0, 1.0, 0.0]),
            "count": 6,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.6666666666666666,
            "weighted_recall": 0.6666666666666666,
        },
    )


def test_recall_result_compute_metrics_multiclass_empty() -> None:
    result = RecallResult(y_true=np.array([]), y_pred=np.array([]), label_type="multiclass")
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "recall": np.array([]),
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_recall_result_compute_metrics_multiclass_prefix_suffix() -> None:
    result = RecallResult(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_recall_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_recall_suffix": 1.0,
            "prefix_micro_recall_suffix": 1.0,
            "prefix_weighted_recall_suffix": 1.0,
        },
    )


def test_recall_result_compute_metrics_multilabel_1_class() -> None:
    result = RecallResult(
        y_true=np.array([[1], [0], [0], [1], [1]]), y_pred=np.array([[1], [0], [0], [1], [1]])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "recall": np.array([1.0]),
            "count": 5,
            "macro_recall": 1.0,
            "micro_recall": 1.0,
            "weighted_recall": 1.0,
        },
    )


def test_recall_result_compute_metrics_multilabel_3_classes() -> None:
    result = RecallResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "recall": np.array([1.0, 1.0, 0.0]),
            "count": 5,
            "macro_recall": 0.6666666666666666,
            "micro_recall": 0.625,
            "weighted_recall": 0.625,
        },
    )


def test_recall_result_compute_metrics_multilabel_empty() -> None:
    result = RecallResult(y_true=np.ones((0, 3)), y_pred=np.ones((0, 3)))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "recall": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_recall": float("nan"),
            "micro_recall": float("nan"),
            "weighted_recall": float("nan"),
        },
        equal_nan=True,
    )


def test_recall_result_compute_metrics_multilabel_prefix_suffix() -> None:
    result = RecallResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_recall_suffix": np.array([1.0, 1.0, 0.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_recall_suffix": 0.6666666666666666,
            "prefix_micro_recall_suffix": 0.625,
            "prefix_weighted_recall_suffix": 0.625,
        },
    )


def test_recall_result_generate_figures() -> None:
    result = RecallResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_recall_result_generate_figures_empty() -> None:
    result = RecallResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
