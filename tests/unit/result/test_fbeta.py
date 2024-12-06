from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import (
    BinaryFbetaScoreResult,
    MulticlassFbetaScoreResult,
    MultilabelFbetaScoreResult,
)

############################################
#     Tests for BinaryFbetaScoreResult     #
############################################


def test_binary_fbeta_score_result_y_true() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_fbeta_score_result_y_true_2d() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_binary_fbeta_score_result_y_pred() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_binary_fbeta_score_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([0, 1, 0, 1, 0, 1]),
    )


def test_binary_fbeta_score_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1, 0])
        )


def test_binary_fbeta_score_result_betas() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]), betas=[0.5, 1, 2]
        ).betas,
        (0.5, 1, 2),
    )


def test_binary_fbeta_score_result_betas_default() -> None:
    assert objects_are_equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).betas,
        (1,),
    )


def test_binary_fbeta_score_result_nan_policy() -> None:
    assert (
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_binary_fbeta_score_result_nan_policy_default() -> None:
    assert (
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).nan_policy
        == "propagate"
    )


def test_binary_fbeta_score_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            nan_policy="incorrect",
        )


def test_binary_fbeta_score_result_repr() -> None:
    assert repr(
        BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryFbetaScoreResult(")


def test_binary_fbeta_score_result_str() -> None:
    assert str(
        BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryFbetaScoreResult(")


def test_binary_fbeta_score_result_equal_true() -> None:
    assert BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )


def test_binary_fbeta_score_result_equal_false_different_y_true() -> None:
    assert not BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1]))
    )


def test_binary_fbeta_score_result_equal_false_different_y_pred() -> None:
    assert not BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0]))
    )


def test_binary_fbeta_score_result_equal_false_different_betas() -> None:
    assert not BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]), betas=[0.5, 1, 2]
        )
    )


def test_binary_fbeta_score_result_equal_false_different_nan_policy() -> None:
    assert not BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    ).equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            nan_policy="omit",
        )
    )


def test_binary_fbeta_score_result_equal_false_different_type() -> None:
    assert not BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_binary_fbeta_score_result_equal_nan_true() -> None:
    assert BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        BinaryFbetaScoreResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_binary_fbeta_score_result_compute_metrics_correct() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "f1": 1.0})


def test_binary_fbeta_score_result_compute_metrics_incorrect() -> None:
    result = BinaryFbetaScoreResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]))
    assert objects_are_allclose(result.compute_metrics(), {"count": 4, "f1": 0.5})


def test_binary_fbeta_score_result_compute_metrics_betas() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        betas=[0.5, 1, 2],
    )
    assert objects_are_equal(
        result.compute_metrics(), {"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0}
    )


def test_binary_fbeta_score_result_compute_metrics_empty() -> None:
    result = BinaryFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "f1": float("nan")}, equal_nan=True
    )


def test_binary_fbeta_score_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_f1_suffix": 1.0},
    )


def test_binary_fbeta_score_result_compute_metrics_binary_nan_omit() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "f1": 1.0})


def test_binary_fbeta_score_result_compute_metrics_binary_nan_propagate() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {"count": 6, "f1": float("nan")},
        equal_nan=True,
    )


def test_binary_fbeta_score_result_compute_metrics_binary_nan_raise() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
        y_pred=np.array([1, 0, 0, 1, 1, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_binary_fbeta_score_result_generate_figures() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_fbeta_score_result_generate_figures_empty() -> None:
    result = BinaryFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_fbeta_score_result_generate_figures_prefix_suffix() -> None:
    result = BinaryFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    )
    assert objects_are_equal(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})


################################################
#     Tests for MulticlassFbetaScoreResult     #
################################################


def test_multiclass_fbeta_score_result_y_true() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_fbeta_score_result_y_true_2d() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_fbeta_score_result_y_pred() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_fbeta_score_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_fbeta_score_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2, 1])
        )


def test_multiclass_fbeta_score_result_betas() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            betas=[0.5, 1, 2],
        ).betas,
        (0.5, 1, 2),
    )


def test_multiclass_fbeta_score_result_betas_default() -> None:
    assert objects_are_equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).betas,
        (1,),
    )


def test_multiclass_fbeta_score_result_nan_policy() -> None:
    assert (
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_multiclass_fbeta_score_result_nan_policy_default() -> None:
    assert (
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).nan_policy
        == "propagate"
    )


def test_multiclass_fbeta_score_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="incorrect",
        )


def test_multiclass_fbeta_score_result_repr() -> None:
    assert repr(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassFbetaScoreResult(")


def test_multiclass_fbeta_score_result_str() -> None:
    assert str(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassFbetaScoreResult(")


def test_multiclass_fbeta_score_result_equal_true() -> None:
    assert MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_fbeta_score_result_equal_false_different_y_true() -> None:
    assert not MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 2, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_fbeta_score_result_equal_false_different_y_pred() -> None:
    assert not MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 3])
        )
    )


def test_multiclass_fbeta_score_result_equal_false_different_betas() -> None:
    assert not MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            betas=[0.5, 1, 2],
        )
    )


def test_multiclass_fbeta_score_result_equal_false_different_nan_policy() -> None:
    assert not MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_pred=np.array([0, 0, 1, 1, 2, 1]),
            nan_policy="omit",
        )
    )


def test_multiclass_fbeta_score_result_equal_false_different_type() -> None:
    assert not MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(42)


def test_multiclass_fbeta_score_result_equal_nan_true() -> None:
    assert MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
    ).equal(
        MulticlassFbetaScoreResult(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        ),
        equal_nan=True,
    )


def test_multiclass_fbeta_score_result_compute_metrics_correct() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_result_compute_metrics_incorrect() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_pred=np.array([0, 0, 1, 1, 1, 1]),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 6,
            "f1": np.array([1.0, 0.6666666666666666, 0.0]),
            "macro_f1": 0.5555555555555555,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.5555555555555555,
        },
    )


def test_multiclass_fbeta_score_result_compute_metrics_betas() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_pred=np.array([0, 0, 1, 1, 2, 2]),
        betas=[0.5, 1, 2],
    )
    assert objects_are_equal(
        result.compute_metrics(),
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


def test_multiclass_fbeta_score_result_compute_metrics_empty() -> None:
    result = MulticlassFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 0,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_result_compute_metrics_prefix_suffix() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_multiclass_fbeta_score_result_compute_metrics_multiclass_nan_omit() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        nan_policy="omit",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 6,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multiclass_fbeta_score_result_compute_metrics_multiclass_nan_propagate() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 7,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_fbeta_score_result_compute_metrics_multiclass_nan_raise() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        y_pred=np.array([0, 1, 1, 2, 2, 2, float("nan")]),
        nan_policy="raise",
    )
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        result.compute_metrics()


def test_multiclass_fbeta_score_result_generate_figures() -> None:
    result = MulticlassFbetaScoreResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_fbeta_score_result_generate_figures_empty() -> None:
    result = MulticlassFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


################################################
#     Tests for MultilabelFbetaScoreResult     #
################################################


def test_multilabel_fbeta_score_result_y_true() -> None:
    assert objects_are_equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_true,
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    )


def test_multilabel_fbeta_score_result_y_pred() -> None:
    assert objects_are_equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_pred,
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )


def test_multilabel_fbeta_score_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1]]),
        )


def test_multilabel_fbeta_score_result_betas() -> None:
    assert objects_are_equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            betas=[0.5, 1, 2],
        ).betas,
        (0.5, 1, 2),
    )


def test_multilabel_fbeta_score_result_betas_default() -> None:
    assert objects_are_equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).betas,
        (1,),
    )


def test_multilabel_fbeta_score_result_nan_policy() -> None:
    assert (
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="omit",
        ).nan_policy
        == "omit"
    )


def test_multilabel_fbeta_score_result_nan_policy_default() -> None:
    assert (
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).nan_policy
        == "propagate"
    )


def test_multilabel_fbeta_score_result_incorrect_nan_policy() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="incorrect",
        )


def test_multilabel_fbeta_score_result_repr() -> None:
    assert repr(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelFbetaScoreResult(")


def test_multilabel_fbeta_score_result_str() -> None:
    assert str(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelFbetaScoreResult(")


def test_multilabel_fbeta_score_result_equal_true() -> None:
    assert MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_fbeta_score_result_equal_false_different_y_true() -> None:
    assert not MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_fbeta_score_result_equal_false_different_y_pred() -> None:
    assert not MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]]),
        )
    )


def test_multilabel_fbeta_score_result_equal_false_different_betas() -> None:
    assert not MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            betas=[0.5, 1, 2],
        )
    )


def test_multilabel_fbeta_score_result_equal_false_different_nan_policy() -> None:
    assert not MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
            nan_policy="omit",
        )
    )


def test_multilabel_fbeta_score_result_equal_false_different_type() -> None:
    assert not MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(42)


def test_multilabel_fbeta_score_result_equal_nan_true() -> None:
    assert MultilabelFbetaScoreResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        MultilabelFbetaScoreResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_multilabel_fbeta_score_result_compute_metrics_1_class_1d() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_result_compute_metrics_1_class_2d() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([[1], [0], [0], [1], [1]]), y_pred=np.array([[1], [0], [0], [1], [1]])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_result_compute_metrics_3_classes() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 5,
            "f1": np.array([1.0, 1.0, 0.0]),
            "macro_f1": 0.6666666666666666,
            "micro_f1": 0.6666666666666666,
            "weighted_f1": 0.625,
        },
    )


def test_multilabel_fbeta_score_result_compute_metrics_betas() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        betas=[0.5, 1, 2],
    )
    assert objects_are_allclose(
        result.compute_metrics(),
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


def test_multilabel_fbeta_score_result_compute_metrics_empty() -> None:
    result = MultilabelFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 0,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_result_compute_metrics_prefix_suffix() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_count_suffix": 5,
            "prefix_macro_f1_suffix": 1.0,
            "prefix_micro_f1_suffix": 1.0,
            "prefix_f1_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_f1_suffix": 1.0,
        },
    )


def test_multilabel_fbeta_score_result_compute_metrics_nan_omit() -> None:
    result = MultilabelFbetaScoreResult(
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
            "count": 5,
            "macro_f1": 1.0,
            "micro_f1": 1.0,
            "f1": np.array([1.0, 1.0, 1.0]),
            "weighted_f1": 1.0,
        },
    )


def test_multilabel_fbeta_score_result_compute_metrics_nan_propagate() -> None:
    result = MultilabelFbetaScoreResult(
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
            "count": 6,
            "macro_f1": float("nan"),
            "micro_f1": float("nan"),
            "f1": np.array([]),
            "weighted_f1": float("nan"),
        },
        equal_nan=True,
    )


def test_multilabel_fbeta_score_result_compute_metrics_nan_raise() -> None:
    result = MultilabelFbetaScoreResult(
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


def test_multilabel_fbeta_score_result_generate_figures() -> None:
    result = MultilabelFbetaScoreResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multilabel_fbeta_score_result_generate_figures_empty() -> None:
    result = MultilabelFbetaScoreResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
