from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal

from arkas.result import AveragePrecisionResult

############################################
#     Tests for AveragePrecisionResult     #
############################################


def test_average_precision_result_y_true() -> None:
    assert objects_are_equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_average_precision_result_y_true_2d() -> None:
    assert objects_are_equal(
        AveragePrecisionResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_score=np.array([[2, -1, 0], [3, 1, 2]]),
        ).y_true,
        np.array([[1, 0, 0], [1, 1, 1]]),
    )


def test_average_precision_result_y_score() -> None:
    assert objects_are_equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ).y_score,
        np.array([2.0, -1.0, 0.0, 3.0, 1.0]),
    )


def test_average_precision_result_y_score_2d() -> None:
    assert objects_are_equal(
        AveragePrecisionResult(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]),
            y_score=np.array([[2, -1, 0], [3, 1, 2]]),
        ).y_score,
        np.array([[2.0, -1.0, 0.0], [3.0, 1.0, 2.0]]),
    )


def test_average_precision_result_label_type() -> None:
    assert (
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        ).label_type
        == "binary"
    )


def test_average_precision_result_y_true_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_true' must be a 1d or 2d array"):
        AveragePrecisionResult(
            y_true=np.ones((2, 3, 4)), y_score=np.ones((2, 3)), label_type="binary"
        )


def test_average_precision_result_y_score_incorrect_ndim() -> None:
    with pytest.raises(ValueError, match="'y_score' must be a 1d or 2d array"):
        AveragePrecisionResult(
            y_true=np.ones((2, 3)), y_score=np.ones((2, 3, 4)), label_type="binary"
        )


def test_average_precision_result_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_score' have different shapes"):
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([1, 0, 0, 1, 1, 0]),
        )


def test_average_precision_result_incorrect_label_type() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([1, 0, 0, 1, 1]),
            label_type="incorrect",
        )


def test_average_precision_result_repr() -> None:
    assert repr(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        )
    ).startswith("AveragePrecisionResult(")


def test_average_precision_result_str() -> None:
    assert str(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        )
    ).startswith("AveragePrecisionResult(")


def test_average_precision_result_equal_true() -> None:
    assert AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
        )
    )


def test_average_precision_result_equal_false_different_y_true() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 0]),
            y_score=np.array([2, -1, 0, 3, 1]),
        )
    )


def test_average_precision_result_equal_false_different_y_score() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 0]))
    )


def test_average_precision_result_equal_false_different_label_type() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([2, -1, 0, 3, 1]),
            label_type="multiclass",
        )
    )


def test_average_precision_result_equal_false_different_type() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(42)


def test_average_precision_result_equal_nan_true() -> None:
    assert AveragePrecisionResult(
        y_true=np.array([1, 0, 0, float("nan"), 1]),
        y_score=np.array([2, -1, 0, 3, float("nan")]),
    ).equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, float("nan"), 1]),
            y_score=np.array([2, -1, 0, 3, float("nan")]),
        ),
        equal_nan=True,
    )


def test_average_precision_result_compute_metrics_binary_correct() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]), label_type="binary"
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "average_precision": 1.0})


def test_average_precision_result_compute_metrics_binary_incorrect() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1]), y_score=np.array([-1, 1, 0, -2]), label_type="binary"
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 4,
            "average_precision": 0.41666666666666663,
        },
    )


def test_average_precision_result_compute_metrics_binary_empty() -> None:
    result = AveragePrecisionResult(y_true=np.array([]), y_score=np.array([]), label_type="binary")
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "average_precision": float("nan")}, equal_nan=True
    )


def test_average_precision_result_compute_metrics_binary_prefix_suffix() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]), label_type="binary"
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_average_precision_suffix": 1.0},
    )


def test_average_precision_result_compute_metrics_multiclass_correct() -> None:
    result = AveragePrecisionResult(
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
        label_type="multiclass",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([1.0, 1.0, 1.0]),
            "count": 6,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_result_compute_metrics_multiclass_incorrect() -> None:
    result = AveragePrecisionResult(
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
        label_type="multiclass",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([0.8333333333333333, 0.75, 0.75]),
            "count": 6,
            "macro_average_precision": 0.7777777777777777,
            "micro_average_precision": 0.75,
            "weighted_average_precision": 0.7777777777777777,
        },
    )


def test_average_precision_result_compute_metrics_multiclass_empty() -> None:
    result = AveragePrecisionResult(
        y_true=np.ones((0,)), y_score=np.ones((0, 3)), label_type="multiclass"
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_result_compute_metrics_multiclass_prefix_suffix() -> None:
    result = AveragePrecisionResult(
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
        label_type="multiclass",
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_average_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 6,
            "prefix_macro_average_precision_suffix": 1.0,
            "prefix_micro_average_precision_suffix": 1.0,
            "prefix_weighted_average_precision_suffix": 1.0,
        },
    )


def test_average_precision_result_compute_metrics_multilabel_1_class() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([[1], [0], [0], [1], [1]]),
        y_score=np.array([[2], [-1], [0], [3], [1]]),
        label_type="multilabel",
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([1.0]),
            "count": 5,
            "macro_average_precision": 1.0,
            "micro_average_precision": 1.0,
            "weighted_average_precision": 1.0,
        },
    )


def test_average_precision_result_compute_metrics_multilabel_3_classes() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
        label_type="multilabel",
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "average_precision": np.array([1.0, 1.0, 0.4777777777777778]),
            "count": 5,
            "macro_average_precision": 0.825925925925926,
            "micro_average_precision": 0.5884199134199134,
            "weighted_average_precision": 0.8041666666666667,
        },
    )


def test_average_precision_result_compute_metrics_multilabel_empty() -> None:
    result = AveragePrecisionResult(
        y_true=np.ones((0, 3)), y_score=np.ones((0, 3)), label_type="multilabel"
    )
    assert objects_are_equal(
        result.compute_metrics(),
        {
            "average_precision": np.array([float("nan"), float("nan"), float("nan")]),
            "count": 0,
            "macro_average_precision": float("nan"),
            "micro_average_precision": float("nan"),
            "weighted_average_precision": float("nan"),
        },
        equal_nan=True,
    )


def test_average_precision_result_compute_metrics_multilabel_prefix_suffix() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
        y_score=np.array([[2, -1, 1], [-1, 1, -2], [0, 2, -3], [3, -2, 4], [1, -3, 5]]),
        label_type="multilabel",
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_average_precision_suffix": np.array([1.0, 1.0, 1.0]),
            "prefix_count_suffix": 5,
            "prefix_macro_average_precision_suffix": 1.0,
            "prefix_micro_average_precision_suffix": 1.0,
            "prefix_weighted_average_precision_suffix": 1.0,
        },
    )


def test_average_precision_result_generate_figures() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]), label_type="binary"
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_average_precision_result_generate_figures_empty() -> None:
    result = AveragePrecisionResult(y_true=np.array([]), y_score=np.array([]), label_type="binary")
    assert objects_are_equal(result.generate_figures(), {})
