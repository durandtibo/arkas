from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.evaluator2 import AccuracyEvaluator, Evaluator
from arkas.state import AccuracyState

#######################################
#     Tests for AccuracyEvaluator     #
#######################################


def test_accuracy_evaluator_repr() -> None:
    assert repr(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyEvaluator(")


def test_accuracy_evaluator_str() -> None:
    assert str(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyEvaluator(")


def test_accuracy_evaluator_equal_true() -> None:
    assert AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_accuracy_evaluator_equal_false_different_state() -> None:
    assert not AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_accuracy_evaluator_equal_false_different_nan_policy() -> None:
    assert not AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            nan_policy="raise",
        )
    )


def test_accuracy_evaluator_equal_false_different_type() -> None:
    assert not AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


def test_accuracy_evaluator_evaluate_binary_correct() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_evaluator_evaluate_binary_incorrect() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1]),
            y_pred=np.array([0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 0.0, "count": 4, "count_correct": 0, "count_incorrect": 4, "error": 1.0},
    )


def test_accuracy_evaluator_evaluate_multiclass_correct() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 6, "count_correct": 6, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_evaluator_evaluate_multiclass_incorrect() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([0, 0, 1, 1, 2]),
            y_pred=np.array([0, 0, 1, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {"accuracy": 0.8, "count": 5, "count_correct": 4, "count_incorrect": 1, "error": 0.2},
    )


def test_accuracy_evaluator_evaluate_empty() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 0,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_accuracy_suffix": 1.0,
            "prefix_count_suffix": 5,
            "prefix_count_correct_suffix": 5,
            "prefix_count_incorrect_suffix": 0,
            "prefix_error_suffix": 0.0,
        },
    )


def test_accuracy_evaluator_evaluate_nan_omit() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="omit",
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_evaluator_evaluate_nan_omit_y_true() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="omit",
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_evaluator_evaluate_nan_omit_y_pred() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="omit",
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0},
    )


def test_accuracy_evaluator_evaluate_nan_propagate() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_evaluator_evaluate_nan_propagate_y_true() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_evaluator_evaluate_nan_propagate_y_pred() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "accuracy": float("nan"),
            "count": 6,
            "count_correct": float("nan"),
            "count_incorrect": float("nan"),
            "error": float("nan"),
        },
        equal_nan=True,
    )


def test_accuracy_evaluator_evaluate_nan_raise() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


def test_accuracy_evaluator_evaluate_nan_raise_y_true() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


def test_accuracy_evaluator_evaluate_nan_raise_y_pred() -> None:
    evaluator = AccuracyEvaluator(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        evaluator.evaluate()


def test_accuracy_evaluator_precompute() -> None:
    assert (
        AccuracyEvaluator(
            AccuracyState(
                y_true=np.array([0, 0, 1, 1, 2, 2]),
                y_pred=np.array([0, 0, 1, 1, 2, 2]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
        .precompute()
        .equal(
            Evaluator(
                {
                    "accuracy": 1.0,
                    "count": 6,
                    "count_correct": 6,
                    "count_incorrect": 0,
                    "error": 0.0,
                }
            )
        )
    )
