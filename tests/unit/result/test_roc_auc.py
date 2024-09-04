from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import BinaryRocAucResult, MulticlassRocAucResult

########################################
#     Tests for BinaryRocAucResult     #
########################################


def test_binary_roc_auc_result_y_true() -> None:
    assert objects_are_equal(
        BinaryRocAucResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_roc_auc_result_y_true_2d() -> None:
    assert objects_are_equal(
        BinaryRocAucResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_score=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_binary_roc_auc_result_y_score() -> None:
    assert objects_are_equal(
        BinaryRocAucResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
        ).y_score,
        np.array([2, -1, 0, 3, 1]),
    )


def test_binary_roc_auc_result_y_score_2d() -> None:
    assert objects_are_equal(
        BinaryRocAucResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_score=np.array([[2, -1, 0], [3, 1, -2]])
        ).y_score,
        np.array([2, -1, 0, 3, 1, -2]),
    )


def test_binary_roc_auc_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1, 4]))


def test_binary_roc_auc_result_repr() -> None:
    assert repr(
        BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    ).startswith("BinaryRocAucResult(")


def test_binary_roc_auc_result_str() -> None:
    assert str(
        BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    ).startswith("BinaryRocAucResult(")


def test_binary_roc_auc_result_equal_true() -> None:
    assert BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    )


def test_binary_roc_auc_result_equal_false_different_y_true() -> None:
    assert not BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 0]), y_score=np.array([2, -1, 0, 3, 1]))
    )


def test_binary_roc_auc_result_equal_false_different_y_score() -> None:
    assert not BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 0])))


def test_binary_roc_auc_result_equal_false_different_type() -> None:
    assert not BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(42)


def test_binary_roc_auc_result_equal_nan_true() -> None:
    assert BinaryRocAucResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_score=np.array([2, -1, 0, 3, float("nan")])
    ).equal(
        BinaryRocAucResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_score=np.array([2, -1, 0, 3, float("nan")]),
        ),
        equal_nan=True,
    )


def test_binary_roc_auc_result_compute_metrics_correct() -> None:
    result = BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "roc_auc": 1.0})


def test_binary_roc_auc_result_compute_metrics_incorrect() -> None:
    result = BinaryRocAucResult(y_true=np.array([1, 0, 0, 1]), y_score=np.array([-1, 1, 0, -2]))
    assert objects_are_allclose(result.compute_metrics(), {"count": 4, "roc_auc": 0.0})


def test_binary_roc_auc_result_compute_metrics_empty() -> None:
    result = BinaryRocAucResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "roc_auc": float("nan")}, equal_nan=True
    )


def test_binary_roc_auc_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_roc_auc_suffix": 1.0},
    )


def test_binary_roc_auc_result_generate_figures() -> None:
    result = BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_roc_auc_result_generate_figures_empty() -> None:
    result = BinaryRocAucResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_roc_auc_result_generate_figures_prefix_suffix() -> None:
    result = BinaryRocAucResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


############################################
#     Tests for MulticlassRocAucResult     #
############################################


def test_multiclass_roc_auc_result_y_true() -> None:
    assert objects_are_equal(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_roc_auc_result_y_true_2d() -> None:
    assert objects_are_equal(
        MulticlassRocAucResult(
            y_true=np.array([[0], [0], [1], [1], [2], [2]]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_roc_auc_result_y_score() -> None:
    assert objects_are_equal(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ).y_score,
        np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    )


def test_multiclass_roc_auc_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different first dimension"):
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                ]
            ),
        )


def test_multiclass_roc_auc_result_repr() -> None:
    assert repr(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        )
    ).startswith("MulticlassRocAucResult(")


def test_multiclass_roc_auc_result_str() -> None:
    assert str(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        )
    ).startswith("MulticlassRocAucResult(")


def test_multiclass_roc_auc_result_equal_true() -> None:
    assert MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    ).equal(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        )
    )


def test_multiclass_roc_auc_result_equal_false_different_y_true() -> None:
    assert not MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    ).equal(
        MulticlassRocAucResult(y_true=np.array([1, 0, 0, 1, 0]), y_score=np.array([2, -1, 0, 3, 1]))
    )


def test_multiclass_roc_auc_result_equal_false_different_y_score() -> None:
    assert not MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    ).equal(
        MulticlassRocAucResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 0]))
    )


def test_multiclass_roc_auc_result_equal_false_different_type() -> None:
    assert not MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    ).equal(42)


def test_multiclass_roc_auc_result_equal_nan_true() -> None:
    assert MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, float("nan")]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, float("nan")],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    ).equal(
        MulticlassRocAucResult(
            y_true=np.array([0, 0, 1, 1, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, float("nan")],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        equal_nan=True,
    )


def test_multiclass_roc_auc_result_compute_metrics_correct() -> None:
    result = MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "roc_auc": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_roc_auc": 1.0,
            "micro_roc_auc": 1.0,
            "weighted_roc_auc": 1.0,
        },
    )


def test_multiclass_roc_auc_result_compute_metrics_incorrect() -> None:
    result = MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.3, 0.5],
                [0.4, 0.4, 0.2],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 6,
            "macro_roc_auc": 0.8333333333333334,
            "micro_roc_auc": 0.826388888888889,
            "roc_auc": np.array([0.9375, 0.8125, 0.75]),
            "weighted_roc_auc": 0.8333333333333334,
        },
    )


def test_multiclass_roc_auc_result_compute_metrics_empty() -> None:
    result = MulticlassRocAucResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "roc_auc": np.array([]),
            "count": 0,
            "macro_roc_auc": float("nan"),
            "micro_roc_auc": float("nan"),
            "weighted_roc_auc": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_roc_auc_result_compute_metrics_prefix_suffix() -> None:
    result = MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_roc_auc_suffix": 1.0,
            "prefix_micro_roc_auc_suffix": 1.0,
            "prefix_weighted_roc_auc_suffix": 1.0,
        },
    )


def test_multiclass_roc_auc_result_generate_figures() -> None:
    result = MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_roc_auc_result_generate_figures_empty() -> None:
    result = MulticlassRocAucResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_roc_auc_result_generate_figures_prefix_suffix() -> None:
    result = MulticlassRocAucResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_score=np.array(
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.3, 0.3],
                [0.1, 0.8, 0.1],
                [0.2, 0.5, 0.3],
                [0.3, 0.2, 0.5],
                [0.1, 0.2, 0.7],
            ]
        ),
    )
    assert objects_are_equal(result.generate_figures(), {})


# ############################################
# #     Tests for MultilabelRocAucResult     #
# ############################################
#
#
# def test_multilabel_roc_auc_result_y_true() -> None:
#     assert objects_are_equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         ).y_true,
#         np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#     )
#
#
# def test_multilabel_roc_auc_result_y_score() -> None:
#     assert objects_are_equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         ).y_score,
#         np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     )
#
#
# def test_multilabel_roc_auc_result_incorrect_shape() -> None:
#     with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1], [1, 1, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         )
#
#
# def test_multilabel_roc_auc_result_repr() -> None:
#     assert repr(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         )
#     ).startswith("MultilabelRocAucResult(")
#
#
# def test_multilabel_roc_auc_result_str() -> None:
#     assert str(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         )
#     ).startswith("MultilabelRocAucResult(")
#
#
# def test_multilabel_roc_auc_result_equal_true() -> None:
#     assert MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     ).equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         )
#     )
#
#
# def test_multilabel_roc_auc_result_equal_false_different_y_true() -> None:
#     assert not MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     ).equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#         )
#     )
#
#
# def test_multilabel_roc_auc_result_equal_false_different_y_score() -> None:
#     assert not MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     ).equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [5, 5, 5]]),
#         )
#     )
#
#
# def test_multilabel_roc_auc_result_equal_false_different_type() -> None:
#     assert not MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     ).equal(42)
#
#
# def test_multilabel_roc_auc_result_equal_nan_true() -> None:
#     assert MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
#         y_score=np.array([[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     ).equal(
#         MultilabelRocAucResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, float("nan")]]),
#             y_score=np.array(
#                 [[float("nan"), -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]
#             ),
#         ),
#         equal_nan=True,
#     )
#
#
# def test_multilabel_roc_auc_result_compute_metrics_1_class_1d() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([1, 0, 0, 1, 1]),
#         y_score=np.array([2, -1, 0, 3, 1]),
#     )
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "roc_auc": np.array([1.0]),
#             "count": 5,
#             "macro_roc_auc": 1.0,
#             "micro_roc_auc": 1.0,
#             "weighted_roc_auc": 1.0,
#         },
#     )
#
#
# def test_multilabel_roc_auc_result_compute_metrics_1_class_2d() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([[1], [0], [0], [1], [1]]),
#         y_score=np.array([[2], [-1], [0], [3], [1]]),
#     )
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "roc_auc": np.array([1.0]),
#             "count": 5,
#             "macro_roc_auc": 1.0,
#             "micro_roc_auc": 1.0,
#             "weighted_roc_auc": 1.0,
#         },
#     )
#
#
# def test_multilabel_roc_auc_result_compute_metrics_incorrect() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
#     )
#     assert objects_are_allclose(
#         result.compute_metrics(),
#         {
#             "count": 5,
#             "macro_roc_auc": 0.6666666666666666,
#             "micro_roc_auc": 0.5446428571428571,
#             "roc_auc": np.array([1.0, 1.0, 0.0]),
#             "weighted_roc_auc": 0.625,
#         },
#     )
#
#
# def test_multilabel_roc_auc_result_compute_metrics_empty() -> None:
#     result = MultilabelRocAucResult(y_true=np.array([]), y_score=np.array([]))
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "roc_auc": np.array([]),
#             "count": 0,
#             "macro_roc_auc": float("nan"),
#             "micro_roc_auc": float("nan"),
#             "weighted_roc_auc": float("nan"),
#         },
#         equal_nan=True,
#     )
#
#
# def test_multilabel_roc_auc_result_compute_metrics_prefix_suffix() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     )
#     assert objects_are_equal(
#         result.compute_metrics(prefix="prefix_", suffix="_suffix"),
#         {
#             "prefix_roc_auc_suffix": np.array([1.0, 1.0, 1.0]),
#             "prefix_count_suffix": 5,
#             "prefix_macro_roc_auc_suffix": 1.0,
#             "prefix_micro_roc_auc_suffix": 1.0,
#             "prefix_weighted_roc_auc_suffix": 1.0,
#         },
#     )
#
#
# def test_multilabel_roc_auc_result_generate_figures() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     )
#     assert objects_are_equal(result.generate_figures(), {})
#
#
# def test_multilabel_roc_auc_result_generate_figures_empty() -> None:
#     result = MultilabelRocAucResult(y_true=np.array([]), y_score=np.array([]))
#     assert objects_are_equal(result.generate_figures(), {})
#
#
# def test_multilabel_roc_auc_result_generate_figures_prefix_suffix() -> None:
#     result = MultilabelRocAucResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
#     )
#     assert objects_are_equal(result.generate_figures(), {})
