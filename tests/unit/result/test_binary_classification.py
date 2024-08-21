from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

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
        np.array([2.0, -1.0, 0.0, 3.0, 1.0], dtype=np.float64),
    )


def test_binary_classification_result_y_score_none() -> None:
    assert (
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_score
        is None
    )


def test_binary_classification_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_binary_classification_result_y_score_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_score' have different shapes"):
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([1, 0, 0, 1, 1, 0]),
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
        f1_betas=[1, 2],
    ).equal(
        BinaryClassificationResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            f1_betas=[1, 2],
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
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]), f1_betas=[1, 2]
        )
    )


def test_binary_classification_result_equal_false_different_type() -> None:
    assert not BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_binary_classification_result_compute_metrics_correct() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "f1": 1.0,
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
            "count": 4,
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
    )


def test_binary_classification_result_compute_metrics_without_y_score() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "f1": 1.0,
        },
    )


def test_binary_classification_result_compute_metrics_beta() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
        f1_betas=[0.5, 1, 2],
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
            "false_negative_rate": 0.0,
            "false_negative": 0,
            "false_positive_rate": 0.0,
            "false_positive": 0,
            "true_negative_rate": 1.0,
            "true_negative": 2,
            "true_positive_rate": 1.0,
            "true_positive": 3,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
            "f0.5": 1.0,
            "f1": 1.0,
            "f2": 1.0,
            "average_precision": 1.0,
            "roc_auc": 1.0,
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
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_false_negative_rate_suffix": 0.0,
            "prefix_false_negative_suffix": 0,
            "prefix_false_positive_rate_suffix": 0.0,
            "prefix_false_positive_suffix": 0,
            "prefix_true_negative_rate_suffix": 1.0,
            "prefix_true_negative_suffix": 2,
            "prefix_true_positive_rate_suffix": 1.0,
            "prefix_true_positive_suffix": 3,
            "prefix_accuracy_suffix": 1.0,
            "prefix_balanced_accuracy_suffix": 1.0,
            "prefix_precision_suffix": 1.0,
            "prefix_recall_suffix": 1.0,
            "prefix_jaccard_suffix": 1.0,
            "prefix_f1_suffix": 1.0,
            "prefix_average_precision_suffix": 1.0,
            "prefix_roc_auc_suffix": 1.0,
        },
    )


def test_binary_classification_result_compute_base_metrics() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_base_metrics(),
        {
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "jaccard": 1.0,
        },
    )


def test_binary_classification_result_compute_confmat_metrics() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_confmat_metrics(),
        {
            "count": 5,
            "count_correct": 5,
            "count_incorrect": 0,
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


def test_binary_classification_result_compute_fbeta_metrics() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(result.compute_fbeta_metrics(), {"f1": 1.0})


def test_binary_classification_result_compute_fbeta_metrics_betas() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
        f1_betas=[0.5, 1, 2],
    )
    assert objects_are_equal(result.compute_fbeta_metrics(), {"f0.5": 1.0, "f1": 1.0, "f2": 1.0})


def test_binary_classification_result_compute_rank_metrics() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(
        result.compute_rank_metrics(), {"average_precision": 1.0, "roc_auc": 1.0}
    )


def test_binary_classification_result_generate_figures() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_score=np.array([2, -1, 0, 3, 1]),
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_classification_result_generate_figures_empty() -> None:
    result = BinaryClassificationResult(
        y_true=np.array([]), y_pred=np.array([]), y_score=np.array([])
    )
    assert objects_are_equal(result.generate_figures(), {})
