from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.result import (
    BinaryConfusionMatrixResult,
    MulticlassConfusionMatrixResult,
    MultilabelConfusionMatrixResult,
)

#################################################
#     Tests for BinaryConfusionMatrixResult     #
#################################################


def test_binary_confusion_matrix_result_y_true() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_confusion_matrix_result_y_true_2d() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_binary_confusion_matrix_result_y_pred() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_binary_confusion_matrix_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([0, 1, 0, 1, 0, 1]),
    )


def test_binary_confusion_matrix_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1, 0])
        )


def test_binary_confusion_matrix_result_nan_policy() -> None:
    assert (
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_binary_confusion_matrix_result_nan_policy_default() -> None:
    assert (
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).nan_policy
        == "propagate"
    )


def test_binary_confusion_matrix_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="incorrect",
        )


def test_binary_confusion_matrix_result_repr() -> None:
    assert repr(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    ).startswith("BinaryConfusionMatrixResult(")


def test_binary_confusion_matrix_result_str() -> None:
    assert str(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    ).startswith("BinaryConfusionMatrixResult(")


def test_binary_confusion_matrix_result_equal_true() -> None:
    assert BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )


def test_binary_confusion_matrix_result_equal_false_different_y_true() -> None:
    assert not BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )


def test_binary_confusion_matrix_result_equal_false_different_y_pred() -> None:
    assert not BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])
        )
    )


def test_binary_confusion_matrix_result_equal_false_different_nan_policy() -> None:
    assert not BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            nan_policy="omit",
        )
    )


def test_binary_confusion_matrix_result_equal_false_different_type() -> None:
    assert not BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_binary_confusion_matrix_result_equal_nan_true() -> None:
    assert BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        BinaryConfusionMatrixResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_binary_confusion_matrix_result_compute_metrics_correct() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
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


def test_binary_confusion_matrix_result_compute_metrics_incorrect() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 1, 0, 0])
    )
    assert objects_are_equal(
        result.compute_metrics(),
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


def test_binary_confusion_matrix_result_compute_metrics_empty() -> None:
    result = BinaryConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
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


def test_binary_confusion_matrix_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
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


def test_binary_confusion_matrix_result_compute_metrics_binary_nan_omit() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
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


def test_binary_confusion_matrix_result_compute_metrics_binary_nan_propagate() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array(
                [[float("nan"), float("nan")], [float("nan"), float("nan")]]
            ),
            "count": 6,
            "false_negative_rate": float("nan"),
            "false_negative": float("nan"),
            "false_positive_rate": float("nan"),
            "false_positive": float("nan"),
            "true_negative_rate": float("nan"),
            "true_negative": float("nan"),
            "true_positive_rate": float("nan"),
            "true_positive": float("nan"),
        },
        equal_nan=True,
    )


def test_binary_confusion_matrix_result_compute_metrics_binary_nan_raise() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_binary_confusion_matrix_result_generate_figures() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_confusion_matrix_result_generate_figures_empty() -> None:
    result = BinaryConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_confusion_matrix_result_generate_figures_prefix_suffix() -> None:
    result = BinaryConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    )
    assert objects_are_equal(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})


#####################################################
#     Tests for MulticlassConfusionMatrixResult     #
#####################################################


def test_multiclass_confusion_matrix_result_y_true() -> None:
    assert objects_are_equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_confusion_matrix_result_y_true_2d() -> None:
    assert objects_are_equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_confusion_matrix_result_y_pred() -> None:
    assert objects_are_equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_confusion_matrix_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_confusion_matrix_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2, 1])
        )


def test_multiclass_confusion_matrix_result_nan_policy() -> None:
    assert (
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_multiclass_confusion_matrix_result_nan_policy_default() -> None:
    assert (
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).nan_policy
        == "propagate"
    )


def test_multiclass_confusion_matrix_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="incorrect",
        )


def test_multiclass_confusion_matrix_result_repr() -> None:
    assert repr(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassConfusionMatrixResult(")


def test_multiclass_confusion_matrix_result_str() -> None:
    assert str(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassConfusionMatrixResult(")


def test_multiclass_confusion_matrix_result_equal_true() -> None:
    assert MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_confusion_matrix_result_equal_false_different_y_true() -> None:
    assert not MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 2, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_confusion_matrix_result_equal_false_different_y_pred() -> None:
    assert not MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 3])
        )
    )


def test_multiclass_confusion_matrix_result_equal_false_different_nan_policy() -> None:
    assert not MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="omit",
        )
    )


def test_multiclass_confusion_matrix_result_equal_false_different_type() -> None:
    assert not MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(42)


def test_multiclass_confusion_matrix_result_equal_nan_true() -> None:
    assert MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
    ).equal(
        MulticlassConfusionMatrixResult(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        ),
        equal_nan=True,
    )


def test_multiclass_confusion_matrix_result_compute_metrics_correct() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 1, 2, 2, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_incorrect() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 2, 0, 1, 2]), y_pred=np.array([0, 1, 1, 2, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[1, 0, 1], [0, 1, 1], [0, 1, 1]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_empty() -> None:
    result = MulticlassConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.zeros((0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_prefix_suffix() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 1, 2, 2, 2]),
        y_pred=np.array([0, 1, 1, 2, 2, 2]),
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_confusion_matrix_suffix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "prefix_count_suffix": 6,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_multiclass_nan_omit() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
            "count": 6,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_multiclass_nan_propagate() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.zeros((0, 0), dtype=np.int64),
            "count": 7,
        },
    )


def test_multiclass_confusion_matrix_result_compute_metrics_multiclass_nan_raise() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_multiclass_confusion_matrix_result_generate_figures() -> None:
    result = MulticlassConfusionMatrixResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_confusion_matrix_result_generate_figures_empty() -> None:
    result = MulticlassConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


#####################################################
#     Tests for MultilabelConfusionMatrixResult     #
#####################################################


def test_multilabel_confusion_matrix_result_y_true() -> None:
    assert objects_are_equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_true,
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    )


def test_multilabel_confusion_matrix_result_y_pred() -> None:
    assert objects_are_equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_pred,
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )


def test_multilabel_confusion_matrix_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1]]),
        )


def test_multilabel_confusion_matrix_result_nan_policy() -> None:
    assert (
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_multilabel_confusion_matrix_result_nan_policy_default() -> None:
    assert (
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).nan_policy
        == "propagate"
    )


def test_multilabel_confusion_matrix_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="incorrect",
        )


def test_multilabel_confusion_matrix_result_repr() -> None:
    assert repr(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelConfusionMatrixResult(")


def test_multilabel_confusion_matrix_result_str() -> None:
    assert str(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelConfusionMatrixResult(")


def test_multilabel_confusion_matrix_result_equal_true() -> None:
    assert MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_confusion_matrix_result_equal_false_different_y_true() -> None:
    assert not MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_confusion_matrix_result_equal_false_different_y_pred() -> None:
    assert not MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]]),
        )
    )


def test_multilabel_confusion_matrix_result_equal_false_different_nan_policy() -> None:
    assert not MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="omit",
        )
    )


def test_multilabel_confusion_matrix_result_equal_false_different_type() -> None:
    assert not MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(42)


def test_multilabel_confusion_matrix_result_equal_nan_true() -> None:
    assert MultilabelConfusionMatrixResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        MultilabelConfusionMatrixResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_multilabel_confusion_matrix_result_compute_metrics_1_class_1d() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_1_class_2d() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array([[1], [0], [0], [1], [1]]), y_pred=np.array([[1], [0], [0], [1], [1]])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_3_classes() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[0, 2], [3, 0]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_empty() -> None:
    result = MultilabelConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.zeros((0, 0, 0), dtype=np.int64),
            "count": 0,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_prefix_suffix() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_confusion_matrix_suffix": np.array(
                [[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]
            ),
            "prefix_count_suffix": 5,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_nan_omit() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
        ),
        y_pred=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
        ),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[[2, 0], [0, 3]], [[3, 0], [0, 2]], [[2, 0], [0, 3]]]),
            "count": 5,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_nan_propagate() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
        ),
        y_pred=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
        ),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.zeros((0, 0, 0), dtype=np.int64),
            "count": 6,
        },
    )


def test_multilabel_confusion_matrix_result_compute_metrics_nan_raise() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [float("nan"), 0, 1]]
        ),
        y_pred=np.array(
            [[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 0, float("nan")]]
        ),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_multilabel_confusion_matrix_result_generate_figures() -> None:
    result = MultilabelConfusionMatrixResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multilabel_confusion_matrix_result_generate_figures_empty() -> None:
    result = MultilabelConfusionMatrixResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
