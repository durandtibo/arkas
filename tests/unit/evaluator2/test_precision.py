from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.evaluator2 import Evaluator, PrecisionEvaluator
from arkas.state import PrecisionRecallState

########################################
#     Tests for PrecisionEvaluator     #
########################################


def test_precision_evaluator_repr() -> None:
    assert repr(
        PrecisionEvaluator(
            PrecisionRecallState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("PrecisionEvaluator(")


def test_precision_evaluator_str() -> None:
    assert str(
        PrecisionEvaluator(
            PrecisionRecallState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("PrecisionEvaluator(")


def test_precision_evaluator_equal_true() -> None:
    assert PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        PrecisionEvaluator(
            PrecisionRecallState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_precision_evaluator_equal_false_different_state() -> None:
    assert not PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        PrecisionEvaluator(
            PrecisionRecallState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_precision_evaluator_equal_false_different_type() -> None:
    assert not PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


def test_precision_evaluator_evaluate_binary_correct() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="binary",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"count": 5, "precision": 1.0},
    )


def test_precision_evaluator_evaluate_binary_incorrect() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1]),
            y_pred=np.array([0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="binary",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"count": 4, "precision": 0.0},
    )


def test_precision_evaluator_evaluate_multiclass_correct() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 2]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "count": 6,
            "macro_precision": 1.0,
            "micro_precision": 1.0,
            "precision": np.array([1.0, 1.0, 1.0]),
            "weighted_precision": 1.0,
        },
    )


def test_precision_evaluator_evaluate_multiclass_incorrect() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multiclass",
        ),
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "count": 6,
            "macro_precision": 0.5,
            "micro_precision": 0.6666666666666666,
            "precision": np.array([1.0, 0.5, 0.0]),
            "weighted_precision": 0.5,
        },
    )


def test_precision_evaluator_evaluate_multilabel() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            y_true_name="target",
            y_pred_name="pred",
            label_type="multilabel",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {
            "count": 5,
            "macro_precision": 0.6666666666666666,
            "micro_precision": 0.7142857142857143,
            "precision": np.array([1.0, 1.0, 0.0]),
            "weighted_precision": 0.625,
        },
    )


def test_precision_evaluator_evaluate_empty() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(), {"count": 0, "precision": float("nan")}, equal_nan=True
    )


def test_precision_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_precision_suffix": 1.0},
    )


def test_precision_evaluator_evaluate_nan_omit() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"count": 5, "precision": 1.0})


def test_precision_evaluator_evaluate_nan_omit_y_true() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"count": 5, "precision": 1.0})


def test_precision_evaluator_evaluate_nan_omit_y_pred() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="omit",
        ),
    )
    assert objects_are_equal(evaluator.evaluate(), {"count": 5, "precision": 1.0})


def test_precision_evaluator_evaluate_nan_propagate() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(), {"count": 6, "precision": float("nan")}, equal_nan=True
    )


def test_precision_evaluator_evaluate_nan_propagate_y_true() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(), {"count": 6, "precision": float("nan")}, equal_nan=True
    )


def test_precision_evaluator_evaluate_nan_propagate_y_pred() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    )
    assert objects_are_equal(
        evaluator.evaluate(), {"count": 6, "precision": float("nan")}, equal_nan=True
    )


def test_precision_evaluator_evaluate_nan_raise() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


def test_precision_evaluator_evaluate_nan_raise_y_true() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([1, 0, 0, 1, 1, 0]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        evaluator.evaluate()


def test_precision_evaluator_evaluate_nan_raise_y_pred() -> None:
    evaluator = PrecisionEvaluator(
        PrecisionRecallState(
            y_true=np.array([1, 0, 0, 1, 1, 0]),
            y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy="raise",
        ),
    )
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        evaluator.evaluate()


def test_precision_evaluator_compute() -> None:
    assert (
        PrecisionEvaluator(
            PrecisionRecallState(
                y_true=np.array([0, 0, 1, 1, 2, 2]),
                y_pred=np.array([0, 0, 1, 1, 2, 2]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
        .compute()
        .equal(
            Evaluator(
                {
                    "count": 6,
                    "macro_precision": 1.0,
                    "micro_precision": 1.0,
                    "precision": np.array([1.0, 1.0, 1.0]),
                    "weighted_precision": 1.0,
                }
            )
        )
    )
