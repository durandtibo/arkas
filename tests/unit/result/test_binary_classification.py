from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal
from matplotlib import pyplot as plt

from arkas.result import BinaryClassificationResult

################################################
#     Tests for BinaryClassificationResult     #
################################################


def test_binary_classification_result_y_true() -> None:
    assert objects_are_equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_classification_result_y_pred() -> None:
    assert objects_are_equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_binary_classification_result_y_score() -> None:
    assert objects_are_equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ).y_score,
        np.array([2, -1, 0, 3, 1]),
    )


def test_binary_classification_result_y_score_none() -> None:
    assert (
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_score
        is None
    )


def test_binary_classification_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_binary_classification_result_y_score_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_binary_classification_result_nan_policy() -> None:
    assert (
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]), nan_policy="omit"
        ).nan_policy
        == "omit"
    )


def test_binary_classification_result_nan_policy_default() -> None:
    assert (
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).nan_policy
        == "propagate"
    )


def test_binary_classification_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="incorrect",
        )


def test_binary_classification_result_repr() -> None:
    assert repr(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
        )
    ).startswith("BinaryClassificationResult(")


def test_binary_classification_result_str() -> None:
    assert str(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
        )
    ).startswith("BinaryClassificationResult(")


def test_binary_classification_result_equal_true() -> None:
    assert BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_binary_classification_result_equal_true_betas() -> None:
    assert BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
        betas=[1, 2],
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            betas=[1, 2],
        )
    )


def test_binary_classification_result_equal_false_different_y_true() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])
        )
    )


def test_binary_classification_result_equal_false_different_y_pred() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])
        )
    )


def test_binary_classification_result_equal_false_different_y_score() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([1, 2, 3, 4, 5]),
        )
    )


def test_binary_classification_result_equal_false_betas() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), betas=[1, 2]
        )
    )


def test_binary_classification_result_equal_false_different_nan_policy() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), nan_policy="omit"
        )
    )


def test_binary_classification_result_equal_false_different_type() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_binary_classification_result_equal_nan_true() -> None:
    assert BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, float("nan"), 1]),
        y_score=np.array([2, -1, float("nan"), 3, 1]),
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, float("nan"), 1]),
            y_score=np.array([2, -1, float("nan"), 3, 1]),
        ),
        equal_nan=True,
    )


def test_binary_classification_result_equal_nan_false() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, float("nan"), 1]),
        y_score=np.array([2, -1, float("nan"), 3, 1]),
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, float("nan"), 1]),
            y_score=np.array([2, -1, float("nan"), 3, 1]),
        )
    )


def test_binary_classification_result_compute_metrics_correct() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": 1.0,
            "count_correct": 5,
            "count_incorrect": 0,
            "count": 5,
            "error": 0.0,
            "balanced_accuracy": 1.0,
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_incorrect() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1]),
        y_pred=np.array([0, 1, 1, 0]),
        y_score=np.array([-1, 1, 0, -2]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "confusion_matrix": np.array([[0, 2], [2, 0]]),
            "count": 4,
            "error": 1.0,
            "count_correct": 0,
            "count_incorrect": 4,
            "false_negative_rate": 1.0,
            "false_negative": 2,
            "false_positive_rate": 1.0,
            "false_positive": 2,
            "true_negative_rate": 0.0,
            "true_negative": 0,
            "true_positive_rate": 0.0,
            "true_positive": 0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "jaccard": 0.0,
            "f1": 0.0,
            "average_precision": 0.41666666666666663,
            "roc_auc": 0.0,
        },
        show_difference=True,
    )


def test_binary_classification_result_compute_metrics_without_y_score() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": 1.0,
            "count_correct": 5,
            "count_incorrect": 0,
            "count": 5,
            "error": 0.0,
            "balanced_accuracy": 1.0,
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "f1": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_beta() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
        betas=[0.5, 1, 2],
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": 1.0,
            "average_precision": 1.0,
            "balanced_accuracy": 1.0,
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "error": 0.0,
            "f0.5": 1.0,
            "f1": 1.0,
            "f2": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "roc_auc": 1.0,
            "true_negative": 2,
            "true_negative_rate": 1.0,
            "true_positive": 3,
            "true_positive_rate": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_average_precision_suffix": 1.0,
            "prefix_balanced_accuracy_suffix": 1.0,
            "prefix_confusion_matrix_suffix": np.array([[2, 0], [0, 3]]),
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
            "prefix_f1_suffix": 1.0,
            "prefix_false_negative_suffix": 0,
            "prefix_false_negative_rate_suffix": 0.0,
            "prefix_false_positive_suffix": 0,
            "prefix_false_positive_rate_suffix": 0.0,
            "prefix_jaccard_suffix": 1.0,
            "prefix_precision_suffix": 1.0,
            "prefix_recall_suffix": 1.0,
            "prefix_roc_auc_suffix": 1.0,
            "prefix_true_negative_suffix": 2,
            "prefix_true_negative_rate_suffix": 1.0,
            "prefix_true_positive_suffix": 3,
            "prefix_true_positive_rate_suffix": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_nan_omit() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_score=np.array([2, -1, 0, 3, 1, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": 1.0,
            "average_precision": 1.0,
            "balanced_accuracy": 1.0,
            "confusion_matrix": np.array([[2, 0], [0, 3]]),
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "error": 0.0,
            "f1": 1.0,
            "false_negative": 0,
            "false_negative_rate": 0.0,
            "false_positive": 0,
            "false_positive_rate": 0.0,
            "jaccard": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "roc_auc": 1.0,
            "true_negative": 2,
            "true_negative_rate": 1.0,
            "true_positive": 3,
            "true_positive_rate": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_nan_propagate() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_score=np.array([2, -1, 0, 3, 1, float("nan")]),
    )
    assert objects_are_equal(
        dict(sorted(result.compute_metrics().items())),
        {
            "accuracy": float("nan"),
            "average_precision": float("nan"),
            "balanced_accuracy": float("nan"),
            "confusion_matrix": np.array(
                [[float("nan"), float("nan")], [float("nan"), float("nan")]]
            ),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
            "f1": float("nan"),
            "false_negative": float("nan"),
            "false_negative_rate": float("nan"),
            "false_positive": float("nan"),
            "false_positive_rate": float("nan"),
            "jaccard": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "roc_auc": float("nan"),
            "true_negative": float("nan"),
            "true_negative_rate": float("nan"),
            "true_positive": float("nan"),
            "true_positive_rate": float("nan"),
        },
        equal_nan=True,
    )


def test_binary_classification_result_compute_metrics_nan_raise() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_score=np.array([2, -1, 0, 3, 1, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_binary_classification_result_generate_figures() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    figures = result.generate_figures()
    assert isinstance(figures, dict)
    assert len(figures) == 1
    assert isinstance(figures["precision_recall"], plt.Figure)


def test_binary_classification_result_generate_figures_empty() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([]), y_pred=np.array([]), y_score=np.array([])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_classification_result_generate_figures_prefix_suffix() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    figures = result.generate_figures(prefix="prefix_", suffix="_suffix")
    assert isinstance(figures, dict)
    assert len(figures) == 1
    assert isinstance(figures["prefix_precision_recall_suffix"], plt.Figure)
