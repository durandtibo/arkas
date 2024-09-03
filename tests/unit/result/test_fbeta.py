from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import BinaryFbetaResult, MulticlassFbetaResult, MultilabelFbetaResult

#######################################
#     Tests for BinaryFbetaResult     #
#######################################


def test_binary_fbeta_result_y_true() -> None:
    assert objects_are_equal(
        BinaryFbetaResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_fbeta_result_y_true_2d() -> None:
    assert objects_are_equal(
        BinaryFbetaResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_binary_fbeta_result_y_pred() -> None:
    assert objects_are_equal(
        BinaryFbetaResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_binary_fbeta_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BinaryFbetaResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([0, 1, 0, 1, 0, 1]),
    )


def test_binary_fbeta_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1, 0]))


def test_binary_fbeta_result_repr() -> None:
    assert repr(
        BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryFbetaResult(")


def test_binary_fbeta_result_str() -> None:
    assert str(
        BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryFbetaResult(")


def test_binary_fbeta_result_equal_true() -> None:
    assert BinaryFbetaResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_binary_fbeta_result_equal_false_different_y_true() -> None:
    assert not BinaryFbetaResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_binary_fbeta_result_equal_false_different_y_pred() -> None:
    assert not BinaryFbetaResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_binary_fbeta_result_equal_false_different_type() -> None:
    assert not BinaryFbetaResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_binary_fbeta_result_equal_nan_true() -> None:
    assert BinaryFbetaResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        BinaryFbetaResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_binary_fbeta_result_compute_metrics_correct() -> None:
    result = BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "f1": 1.0})


def test_binary_fbeta_result_compute_metrics_incorrect() -> None:
    result = BinaryFbetaResult(y_true=np.array([1, 0, 0, 1]), y_pred=np.array([1, 0, 1, 0]))
    assert objects_are_allclose(result.compute_metrics(), {"count": 4, "f1": 0.5})


def test_binary_fbeta_result_compute_metrics_betas() -> None:
    result = BinaryFbetaResult(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        betas=[0.5, 1, 2],
    )
    assert objects_are_equal(
        result.compute_metrics(), {"count": 5, "f0.5": 1.0, "f1": 1.0, "f2": 1.0}
    )


def test_binary_fbeta_result_compute_metrics_empty() -> None:
    result = BinaryFbetaResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "f1": float("nan")}, equal_nan=True
    )


def test_binary_fbeta_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_f1_suffix": 1.0},
    )


def test_binary_fbeta_result_generate_figures() -> None:
    result = BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_fbeta_result_generate_figures_empty() -> None:
    result = BinaryFbetaResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_fbeta_result_generate_figures_prefix_suffix() -> None:
    result = BinaryFbetaResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})


###########################################
#     Tests for MulticlassFbetaResult     #
###########################################


def test_multiclass_fbeta_result_y_true() -> None:
    assert objects_are_equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_fbeta_result_y_true_2d() -> None:
    assert objects_are_equal(
        MulticlassFbetaResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_fbeta_result_y_pred() -> None:
    assert objects_are_equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_fbeta_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MulticlassFbetaResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_fbeta_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2, 1])
        )


def test_multiclass_fbeta_result_repr() -> None:
    assert repr(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassFbetaResult(")


def test_multiclass_fbeta_result_str() -> None:
    assert str(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassFbetaResult(")


def test_multiclass_fbeta_result_equal_true() -> None:
    assert MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_fbeta_result_equal_false_different_y_true() -> None:
    assert not MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 2, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_fbeta_result_equal_false_different_y_pred() -> None:
    assert not MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 3])
        )
    )


def test_multiclass_fbeta_result_equal_false_different_type() -> None:
    assert not MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(42)


def test_multiclass_fbeta_result_equal_nan_true() -> None:
    assert MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
    ).equal(
        MulticlassFbetaResult(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        ),
        equal_nan=True,
    )


def test_multiclass_fbeta_result_compute_metrics_correct() -> None:
    result = MulticlassFbetaResult(
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


def test_multiclass_fbeta_result_compute_metrics_incorrect() -> None:
    result = MulticlassFbetaResult(
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


def test_multiclass_fbeta_result_compute_metrics_betas() -> None:
    result = MulticlassFbetaResult(
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


def test_multiclass_fbeta_result_compute_metrics_empty() -> None:
    result = MulticlassFbetaResult(y_true=np.array([]), y_pred=np.array([]))
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


def test_multiclass_fbeta_result_compute_metrics_prefix_suffix() -> None:
    result = MulticlassFbetaResult(
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


def test_multiclass_fbeta_result_generate_figures() -> None:
    result = MulticlassFbetaResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_fbeta_result_generate_figures_empty() -> None:
    result = MulticlassFbetaResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


###########################################
#     Tests for MultilabelFbetaResult     #
###########################################


def test_multilabel_fbeta_result_y_true() -> None:
    assert objects_are_equal(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_true,
        np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
    )


def test_multilabel_fbeta_result_y_pred() -> None:
    assert objects_are_equal(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        ).y_pred,
        np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )


def test_multilabel_fbeta_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1]]),
        )


def test_multilabel_fbeta_result_repr() -> None:
    assert repr(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelFbetaResult(")


def test_multilabel_fbeta_result_str() -> None:
    assert str(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    ).startswith("MultilabelFbetaResult(")


def test_multilabel_fbeta_result_equal_true() -> None:
    assert MultilabelFbetaResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_fbeta_result_equal_false_different_y_true() -> None:
    assert not MultilabelFbetaResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
        )
    )


def test_multilabel_fbeta_result_equal_false_different_y_pred() -> None:
    assert not MultilabelFbetaResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(
        MultilabelFbetaResult(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]]),
        )
    )


def test_multilabel_fbeta_result_equal_false_different_type() -> None:
    assert not MultilabelFbetaResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    ).equal(42)


def test_multilabel_fbeta_result_equal_nan_true() -> None:
    assert MultilabelFbetaResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        MultilabelFbetaResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_multilabel_fbeta_result_compute_metrics_1_class_1d() -> None:
    result = MultilabelFbetaResult(
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


def test_multilabel_fbeta_result_compute_metrics_1_class_2d() -> None:
    result = MultilabelFbetaResult(
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


def test_multilabel_fbeta_result_compute_metrics_3_classes() -> None:
    result = MultilabelFbetaResult(
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


def test_multilabel_fbeta_result_compute_metrics_betas() -> None:
    result = MultilabelFbetaResult(
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


def test_multilabel_fbeta_result_compute_metrics_empty() -> None:
    result = MultilabelFbetaResult(y_true=np.array([]), y_pred=np.array([]))
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


def test_multilabel_fbeta_result_compute_metrics_prefix_suffix() -> None:
    result = MultilabelFbetaResult(
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


def test_multilabel_fbeta_result_generate_figures() -> None:
    result = MultilabelFbetaResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multilabel_fbeta_result_generate_figures_empty() -> None:
    result = MultilabelFbetaResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
