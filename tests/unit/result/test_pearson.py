from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import PearsonCorrelationResult

##############################################
#     Tests for PearsonCorrelationResult     #
##############################################


def test_pearson_correlation_result_x() -> None:
    assert objects_are_allclose(
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1])).x,
        np.array([1, 2, 3, 4, 5]),
    )


def test_pearson_correlation_result_x_2d() -> None:
    assert objects_are_allclose(
        PearsonCorrelationResult(
            x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[6, 5, 4], [3, 2, 1]])
        ).x,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_pearson_correlation_result_y() -> None:
    assert objects_are_allclose(
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1])).y,
        np.array([5, 4, 3, 2, 1]),
    )


def test_pearson_correlation_result_y_2d() -> None:
    assert objects_are_allclose(
        PearsonCorrelationResult(
            x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[6, 5, 4], [3, 2, 1]])
        ).y,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_pearson_correlation_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5, 0]))


def test_pearson_correlation_result_alternative() -> None:
    assert (
        PearsonCorrelationResult(
            x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1]), alternative="less"
        ).alternative
        == "less"
    )


def test_pearson_correlation_result_alternative_default() -> None:
    assert (
        PearsonCorrelationResult(
            x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1])
        ).alternative
        == "two-sided"
    )


def test_pearson_correlation_result_repr() -> None:
    assert repr(
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    ).startswith("PearsonCorrelationResult(")


def test_pearson_correlation_result_str() -> None:
    assert str(
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    ).startswith("PearsonCorrelationResult(")


def test_pearson_correlation_result_equal_true() -> None:
    assert PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])).equal(
        PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    )


def test_pearson_correlation_result_equal_false_different_x() -> None:
    assert not PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])
    ).equal(PearsonCorrelationResult(x=np.array([1, 0, 0, 1, 0]), y=np.array([1, 2, 3, 4, 5])))


def test_pearson_correlation_result_equal_false_different_y() -> None:
    assert not PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])
    ).equal(PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1])))


def test_pearson_correlation_result_equal_false_different_alternative() -> None:
    assert not PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1]), alternative="less"
    ).equal(PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1])))


def test_pearson_correlation_result_equal_false_different_type() -> None:
    assert not PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])
    ).equal(42)


def test_pearson_correlation_result_equal_nan_true() -> None:
    assert PearsonCorrelationResult(
        x=np.array([1, 0, 0, 1, float("nan")]), y=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        PearsonCorrelationResult(
            x=np.array([1, 0, 0, 1, float("nan")]), y=np.array([1, 0, 0, float("nan"), 1])
        ),
        equal_nan=True,
    )


def test_pearson_correlation_result_equal_nan_false() -> None:
    assert not PearsonCorrelationResult(
        x=np.array([1, 0, 0, 1, float("nan")]), y=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        PearsonCorrelationResult(
            x=np.array([1, 0, 0, 1, float("nan")]), y=np.array([1, 0, 0, float("nan"), 1])
        )
    )


def test_pearson_correlation_result_compute_metrics_positive_correlation() -> None:
    result = PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    assert objects_are_allclose(
        result.compute_metrics(), {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0}
    )


def test_pearson_correlation_result_compute_metrics_negative_correlation() -> None:
    result = PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([5, 4, 3, 2, 1]))
    assert objects_are_allclose(
        result.compute_metrics(), {"count": 5, "pearson_coeff": -1.0, "pearson_pvalue": 0.0}
    )


def test_pearson_correlation_result_compute_metrics_empty() -> None:
    result = PearsonCorrelationResult(x=np.array([]), y=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 0, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


def test_pearson_correlation_result_compute_metrics_alternative_less() -> None:
    result = PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]), alternative="less"
    )
    assert objects_are_allclose(
        result.compute_metrics(), {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 1.0}
    )


def test_pearson_correlation_result_compute_metrics_prefix_suffix() -> None:
    result = PearsonCorrelationResult(
        x=np.array([1, 2, 3, 4, 5]),
        y=np.array([1, 2, 3, 4, 5]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_count_suffix": 5,
            "prefix_pearson_coeff_suffix": 1.0,
            "prefix_pearson_pvalue_suffix": 0.0,
        },
    )


def test_pearson_correlation_result_generate_figures() -> None:
    result = PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_pearson_correlation_result_generate_figures_empty() -> None:
    result = PearsonCorrelationResult(x=np.array([]), y=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_pearson_correlation_result_generate_figures_prefix_suffix() -> None:
    result = PearsonCorrelationResult(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]))
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
