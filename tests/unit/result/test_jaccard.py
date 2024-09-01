from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import BinaryJaccardResult, MulticlassJaccardResult

#########################################
#     Tests for BinaryJaccardResult     #
#########################################


def test_binary_jaccard_result_y_true() -> None:
    assert objects_are_equal(
        BinaryJaccardResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_binary_jaccard_result_y_true_2d() -> None:
    assert objects_are_equal(
        BinaryJaccardResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_true,
        np.array([1, 0, 0, 1, 1, 1]),
    )


def test_binary_jaccard_result_y_pred() -> None:
    assert objects_are_equal(
        BinaryJaccardResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        ).y_pred,
        np.array([1, 0, 1, 0, 1]),
    )


def test_binary_jaccard_result_y_pred_2d() -> None:
    assert objects_are_equal(
        BinaryJaccardResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[0, 1, 0], [1, 0, 1]])
        ).y_pred,
        np.array([0, 1, 0, 1, 0, 1]),
    )


def test_binary_jaccard_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1, 0]))


def test_binary_jaccard_result_repr() -> None:
    assert repr(
        BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryJaccardResult(")


def test_binary_jaccard_result_str() -> None:
    assert str(
        BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).startswith("BinaryJaccardResult(")


def test_binary_jaccard_result_equal_true() -> None:
    assert BinaryJaccardResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_binary_jaccard_result_equal_false_different_y_true() -> None:
    assert not BinaryJaccardResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 0]), y_pred=np.array([1, 0, 1, 0, 1])))


def test_binary_jaccard_result_equal_false_different_y_pred() -> None:
    assert not BinaryJaccardResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 0])))


def test_binary_jaccard_result_equal_false_different_type() -> None:
    assert not BinaryJaccardResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
    ).equal(42)


def test_binary_jaccard_result_equal_nan_true() -> None:
    assert BinaryJaccardResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
    ).equal(
        BinaryJaccardResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_pred=np.array([0, 1, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_binary_jaccard_result_compute_metrics_correct() -> None:
    result = BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "jaccard": 1.0})


def test_binary_jaccard_result_compute_metrics_incorrect() -> None:
    result = BinaryJaccardResult(
        y_true=np.array([1, 0, 0, 1, 1, 1]),
        y_pred=np.array([1, 0, 1, 0, 1, 1]),
    )
    assert objects_are_allclose(result.compute_metrics(), {"count": 6, "jaccard": 0.6})


def test_binary_jaccard_result_compute_metrics_empty() -> None:
    result = BinaryJaccardResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "jaccard": float("nan")}, equal_nan=True
    )


def test_binary_jaccard_result_compute_metrics_prefix_suffix() -> None:
    result = BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1]))
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_jaccard_suffix": 1.0},
    )


def test_binary_jaccard_result_generate_figures() -> None:
    result = BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_jaccard_result_generate_figures_empty() -> None:
    result = BinaryJaccardResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


def test_binary_jaccard_result_generate_figures_prefix_suffix() -> None:
    result = BinaryJaccardResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    assert objects_are_equal(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})


#############################################
#     Tests for MulticlassJaccardResult     #
#############################################


def test_multiclass_jaccard_result_y_true() -> None:
    assert objects_are_equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_jaccard_result_y_true_2d() -> None:
    assert objects_are_equal(
        MulticlassJaccardResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_true,
        np.array([0, 0, 1, 1, 2, 2]),
    )


def test_multiclass_jaccard_result_y_pred() -> None:
    assert objects_are_equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_jaccard_result_y_pred_2d() -> None:
    assert objects_are_equal(
        MulticlassJaccardResult(
            y_true=np.array([[0, 0, 1], [1, 2, 2]]), y_pred=np.array([[0, 0, 1], [1, 2, 1]])
        ).y_pred,
        np.array([0, 0, 1, 1, 2, 1]),
    )


def test_multiclass_jaccard_result_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2, 1])
        )


def test_multiclass_jaccard_result_repr() -> None:
    assert repr(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassJaccardResult(")


def test_multiclass_jaccard_result_str() -> None:
    assert str(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
        )
    ).startswith("MulticlassJaccardResult(")


def test_multiclass_jaccard_result_equal_true() -> None:
    assert MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_jaccard_result_equal_false_different_y_true() -> None:
    assert not MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 2, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
        )
    )


def test_multiclass_jaccard_result_equal_false_different_y_pred() -> None:
    assert not MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 3])
        )
    )


def test_multiclass_jaccard_result_equal_false_different_type() -> None:
    assert not MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 1])
    ).equal(42)


def test_multiclass_jaccard_result_equal_nan_true() -> None:
    assert MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
    ).equal(
        MulticlassJaccardResult(
            y_true=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
            y_pred=np.array([0, 0, 1, 1, 2, 2, float("nan")]),
        ),
        equal_nan=True,
    )


def test_multiclass_jaccard_result_compute_metrics_correct() -> None:
    result = MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 6,
            "macro_jaccard": 1.0,
            "micro_jaccard": 1.0,
            "jaccard": np.array([1.0, 1.0, 1.0]),
            "weighted_jaccard": 1.0,
        },
    )


def test_multiclass_jaccard_result_compute_metrics_incorrect() -> None:
    result = MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]),
        y_pred=np.array([0, 0, 1, 1, 1, 1]),
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 6,
            "jaccard": np.array([1.0, 0.5, 0.0]),
            "macro_jaccard": 0.5,
            "micro_jaccard": 0.5,
            "weighted_jaccard": 0.5,
        },
    )


def test_multiclass_jaccard_result_compute_metrics_empty() -> None:
    result = MulticlassJaccardResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "count": 0,
            "macro_jaccard": float("nan"),
            "micro_jaccard": float("nan"),
            "jaccard": np.array([]),
            "weighted_jaccard": float("nan"),
        },
        equal_nan=True,
    )


def test_multiclass_jaccard_result_compute_metrics_prefix_suffix() -> None:
    result = MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_count_suffix": 6,
            "prefix_macro_jaccard_suffix": 1.0,
            "prefix_micro_jaccard_suffix": 1.0,
            "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_weighted_jaccard_suffix": 1.0,
        },
    )


def test_multiclass_jaccard_result_generate_figures() -> None:
    result = MulticlassJaccardResult(
        y_true=np.array([0, 0, 1, 1, 2, 2]), y_pred=np.array([0, 0, 1, 1, 2, 2])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_multiclass_jaccard_result_generate_figures_empty() -> None:
    result = MulticlassJaccardResult(y_true=np.array([]), y_pred=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})


#############################################
#     Tests for MultilabelJaccardResult     #
#############################################


# def test_multilabel_jaccard_result_y_true() -> None:
#     assert objects_are_equal(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         ).y_true,
#         np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#     )
#
#
# def test_multilabel_jaccard_result_y_pred() -> None:
#     assert objects_are_equal(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         ).y_pred,
#         np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     )
#
#
# def test_multilabel_jaccard_result_incorrect_shape() -> None:
#     with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0], [1, 1, 1]]),
#         )
#
#
# def test_multilabel_jaccard_result_repr() -> None:
#     assert repr(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         )
#     ).startswith("MultilabelJaccardResult(")
#
#
# def test_multilabel_jaccard_result_str() -> None:
#     assert str(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         )
#     ).startswith("MultilabelJaccardResult(")
#
#
# def test_multilabel_jaccard_result_equal_true() -> None:
#     assert MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     ).equal(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         )
#     )
#
#
# def test_multilabel_jaccard_result_equal_false_different_y_true() -> None:
#     assert not MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     ).equal(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#         )
#     )
#
#
# def test_multilabel_jaccard_result_equal_false_different_y_pred() -> None:
#     assert not MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     ).equal(
#         MultilabelJaccardResult(
#             y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#             y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 1, 1]]),
#         )
#     )
#
#
# def test_multilabel_jaccard_result_equal_false_different_type() -> None:
#     assert not MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     ).equal(42)
#
#
# def test_multilabel_jaccard_result_equal_nan_true() -> None:
#     assert MultilabelJaccardResult(
#         y_true=np.array([1, 0, 0, float("nan"), 1]), y_pred=np.array([0, 1, 0, float("nan"), 1])
#     ).equal(
#         MultilabelJaccardResult(
#             y_true=np.array([1, 0, 0, float("nan"), 1]),
#             y_pred=np.array([0, 1, 0, float("nan"), 1]),
#         ),
#         equal_nan=True,
#     )
#
#
# def test_multilabel_jaccard_result_compute_metrics_1_class_1d() -> None:
#     result = MultilabelJaccardResult(
#         y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 0, 1, 1])
#     )
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "count": 5,
#             "macro_jaccard": 1.0,
#             "micro_jaccard": 1.0,
#             "jaccard": np.array([1.0]),
#             "weighted_jaccard": 1.0,
#         },
#     )
#
#
# def test_multilabel_jaccard_result_compute_metrics_1_class_2d() -> None:
#     result = MultilabelJaccardResult(
#         y_true=np.array([[1], [0], [0], [1], [1]]), y_pred=np.array([[1], [0], [0], [1], [1]])
#     )
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "count": 5,
#             "macro_jaccard": 1.0,
#             "micro_jaccard": 1.0,
#             "jaccard": np.array([1.0]),
#             "weighted_jaccard": 1.0,
#         },
#     )
#
#
# def test_multilabel_jaccard_result_compute_metrics_3_classes() -> None:
#     result = MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     )
#     assert objects_are_allclose(
#         result.compute_metrics(),
#         {
#             "jaccard": np.array([1.0, 1.0, 0.0]),
#             "count": 5,
#             "macro_jaccard": 0.6666666666666666,
#             "micro_jaccard": 0.5,
#             "weighted_jaccard": 0.625,
#         },
#     )
#
#
# def test_multilabel_jaccard_result_compute_metrics_empty() -> None:
#     result = MultilabelJaccardResult(y_true=np.array([]), y_pred=np.array([]))
#     assert objects_are_equal(
#         result.compute_metrics(),
#         {
#             "count": 0,
#             "macro_jaccard": float("nan"),
#             "micro_jaccard": float("nan"),
#             "jaccard": np.array([]),
#             "weighted_jaccard": float("nan"),
#         },
#         equal_nan=True,
#     )
#
#
# def test_multilabel_jaccard_result_compute_metrics_prefix_suffix() -> None:
#     result = MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#     )
#     assert objects_are_equal(
#         result.compute_metrics(prefix="prefix_", suffix="_suffix"),
#         {
#             "prefix_count_suffix": 5,
#             "prefix_macro_jaccard_suffix": 1.0,
#             "prefix_micro_jaccard_suffix": 1.0,
#             "prefix_jaccard_suffix": np.array([1.0, 1.0, 1.0]),
#             "prefix_weighted_jaccard_suffix": 1.0,
#         },
#     )
#
#
# def test_multilabel_jaccard_result_generate_figures() -> None:
#     result = MultilabelJaccardResult(
#         y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
#         y_pred=np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1], [1, 0, 0], [1, 0, 0]]),
#     )
#     assert objects_are_equal(result.generate_figures(), {})
#
#
# def test_multilabel_jaccard_result_generate_figures_empty() -> None:
#     result = MultilabelJaccardResult(y_true=np.array([]), y_pred=np.array([]))
#     assert objects_are_equal(result.generate_figures(), {})
