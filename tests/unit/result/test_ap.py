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
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
        ).y_true,
        np.array([1, 0, 0, 1, 1]),
    )


def test_average_precision_result_y_score() -> None:
    assert objects_are_equal(
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
        ).y_score,
        np.array([2.0, -1.0, 0.0, 3.0, 1.0]),
    )


def test_average_precision_result_y_score_incorrect_shape() -> None:
    with pytest.raises(ValueError, match="'y_true' and 'y_score' have different shapes"):
        AveragePrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 1, 0])
        )


def test_average_precision_result_repr() -> None:
    assert repr(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    ).startswith("AveragePrecisionResult(")


def test_average_precision_result_str() -> None:
    assert str(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    ).startswith("AveragePrecisionResult(")


def test_average_precision_result_equal_true() -> None:
    assert AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1]))
    )


def test_average_precision_result_equal_false_different_y_true() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 0]), y_score=np.array([2, -1, 0, 3, 1]))
    )


def test_average_precision_result_equal_false_different_y_score() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(
        AveragePrecisionResult(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([1, 0, 0, 1, 0]))
    )


def test_average_precision_result_equal_false_different_type() -> None:
    assert not AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    ).equal(42)


def test_average_precision_result_compute_metrics_correct() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(result.compute_metrics(), {"count": 5, "average_precision": 1.0})


def test_average_precision_result_compute_metrics_incorrect() -> None:
    result = AveragePrecisionResult(y_true=np.array([1, 0, 0, 1]), y_score=np.array([-1, 1, 0, -2]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {
            "count": 4,
            "average_precision": 0.41666666666666663,
        },
    )


def test_average_precision_result_compute_metrics_empty() -> None:
    result = AveragePrecisionResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(
        result.compute_metrics(), {"count": 0, "average_precision": float("nan")}, equal_nan=True
    )


def test_average_precision_result_compute_metrics_prefix_suffix() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_average_precision_suffix": 1.0},
    )


def test_average_precision_result_generate_figures() -> None:
    result = AveragePrecisionResult(
        y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([2, -1, 0, 3, 1])
    )
    assert objects_are_equal(result.generate_figures(), {})


def test_average_precision_result_generate_figures_empty() -> None:
    result = AveragePrecisionResult(y_true=np.array([]), y_score=np.array([]))
    assert objects_are_equal(result.generate_figures(), {})
