from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import AccuracyResult, BalancedAccuracyResult
from arkas.result.accuracy import accuracy_metrics

####################################
#     Tests for AccuracyResult     #
####################################


def test_accuracy_result_y_true() -> None:
    assert objects_are_equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_accuracy_result_y_true_2d() -> None:
    assert objects_are_equal(
        AccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_accuracy_result_y_pred() -> None:
    assert objects_are_equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_accuracy_result_y_pred_2d() -> None:
    assert objects_are_equal(
        AccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_pred,
        np.array([1, 0, 1, 0, 1, 0]),
    )


def test_accuracy_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_pred' have different shapes"):
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0]))


def test_accuracy_result_repr() -> None:
    assert repr(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("AccuracyResult(")


def test_accuracy_result_str() -> None:
    assert str(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("AccuracyResult(")


def test_accuracy_result_equal_true() -> None:
    assert AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])).equal(
        AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_accuracy_result_equal_false_different_y_true() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(AccuracyResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1])))


def test_accuracy_result_equal_false_different_y_pred() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_accuracy_result_equal_false_different_type() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_accuracy_result_equal_nan_true() -> None:
    assert AccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        AccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_accuracy_result_equal_nan_false() -> None:
    assert not AccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        AccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_accuracy_result_compute_metrics_correct() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_result_compute_metrics_incorrect() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_result_compute_metrics_empty() -> None:
    result = AccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": 0,
            "count_incorrect": 0,
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_result_compute_metrics_prefix_suffix() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


def test_accuracy_result_compute_metrics_binary() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 0, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


def test_accuracy_result_compute_metrics_multiclass() -> None:
    result = AccuracyResult(y_true=np.array([0, 1, 2, 3, 4]), y_pred=np.array([0, 1, 1, 3, 3]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"accuracy": 0.6, "count": 5, "count_correct": 3, "count_incorrect": 2, "error": 0.4},
    )


def test_accuracy_result_generate_figures() -> None:
    result = AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_accuracy_result_generate_figures_empty() -> None:
    result = AccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


############################################
#     Tests for BalancedAccuracyResult     #
############################################


def test_balanced_accuracy_result_y_true() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_balanced_accuracy_result_y_true_2d() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_balanced_accuracy_result_y_pred() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_balanced_accuracy_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BalancedAccuracyResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 1], [0, 1, 0]])
        ).y_pred,
        np.array([1, 0, 1, 0, 1, 0]),
    )


def test_balanced_accuracy_result_y_pred_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_pred' have different shapes"):
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1, 0])
        )


def test_balanced_accuracy_result_repr() -> None:
    assert repr(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BalancedAccuracyResult(")


def test_balanced_accuracy_result_str() -> None:
    assert str(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BalancedAccuracyResult(")


def test_balanced_accuracy_result_equal_true() -> None:
    assert BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_balanced_accuracy_result_equal_false_different_y_true() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1]))
    )


def test_balanced_accuracy_result_equal_false_different_y_pred() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0]))
    )


def test_balanced_accuracy_result_equal_false_different_type() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(42)


def test_balanced_accuracy_result_equal_nan_true() -> None:
    assert BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_balanced_accuracy_result_equal_nan_false() -> None:
    assert not BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        BalancedAccuracyResult(
            y_true=np.array([1, 0, 0, 1, float("nan")]), y_pred=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_balanced_accuracy_result_compute_metrics_correct() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 1.0, "count": 5},
    )


def test_balanced_accuracy_result_compute_metrics_incorrect() -> None:
    result = BalancedAccuracyResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 0.0, "count": 4},
    )


def test_balanced_accuracy_result_compute_metrics_empty() -> None:
    result = BalancedAccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": float("nan"), "count": 0},
        equal_nan=True,
    )


def test_balanced_accuracy_result_compute_metrics_prefix_suffix() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_balanced_accuracy_suffix": 1.0, "prefix_count_suffix": 5},
    )


def test_balanced_accuracy_result_compute_metrics_binary() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 0, 1, 0]), y_pred=np.array([1, 0, 0, 1, 1, 1])
    )
    assert objects_are_allclose(result.compute_metrics(), {"balanced_accuracy": 0.75, "count": 6})


def test_balanced_accuracy_result_compute_metrics_multiclass() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([0, 1, 2, 3, 4]), y_pred=np.array([0, 1, 1, 3, 3])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"balanced_accuracy": 0.6, "count": 5},
    )


def test_balanced_accuracy_result_generate_figures() -> None:
    result = BalancedAccuracyResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_balanced_accuracy_result_generate_figures_empty() -> None:
    result = BalancedAccuracyResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


######################################
#     Tests for accuracy_metrics     #
######################################


def test_accuracy_metrics_binary_correct() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_binary_correct_2d() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([[1, 0, 0], [1, 1, 0]]), y_pred=np.array([[1, 0, 0], [1, 1, 0]])
        ),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_binary_incorrect() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([0, 1, 1, 0])),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_metrics_multiclass_correct() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_multiclass_incorrect() -> None:
    assert objects_are_allclose(
        accuracy_metrics(y_true=np.array([0, 0, 1, 1, 2]), y_pred=np.array([0, 0, 1, 1, 1])),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


def test_accuracy_metrics_empty() -> None:
    assert objects_are_equal(
        accuracy_metrics(y_true=np.array([]), y_pred=np.array([])),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": 0,
            "count_incorrect": 0,
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


def test_accuracy_metrics_nans() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_y_true_nan() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_metrics_y_pred_nan() -> None:
    assert objects_are_equal(
        accuracy_metrics(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        ),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )
