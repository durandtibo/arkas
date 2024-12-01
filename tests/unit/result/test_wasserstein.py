from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import WassersteinDistanceResult

###############################################
#     Tests for WassersteinDistanceResult     #
###############################################


def test_wasserstein_distance_result_u_values() -> None:
    assert objects_are_allclose(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1])
        ).u_values,
        np.array([1, 2, 3, 4, 5]),
    )


def test_wasserstein_distance_result_u_values_2d() -> None:
    assert objects_are_allclose(
        WassersteinDistanceResult(
            u_values=np.array([[1, 2, 3], [4, 5, 6]]), v_values=np.array([[6, 5, 4], [3, 2, 1]])
        ).u_values,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_wasserstein_distance_result_v_values() -> None:
    assert objects_are_allclose(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1])
        ).v_values,
        np.array([5, 4, 3, 2, 1]),
    )


def test_wasserstein_distance_result_v_values_2d() -> None:
    assert objects_are_allclose(
        WassersteinDistanceResult(
            u_values=np.array([[1, 2, 3], [4, 5, 6]]), v_values=np.array([[6, 5, 4], [3, 2, 1]])
        ).v_values,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_wasserstein_distance_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5, 0])
        )


def test_wasserstein_distance_result_repr() -> None:
    assert repr(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("WassersteinDistanceResult(")


def test_wasserstein_distance_result_str() -> None:
    assert str(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
        )
    ).startswith("WassersteinDistanceResult(")


def test_wasserstein_distance_result_equal_true() -> None:
    assert WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
        )
    )


def test_wasserstein_distance_result_equal_false_different_x() -> None:
    assert not WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        WassersteinDistanceResult(
            u_values=np.array([1, 0, 0, 1, 0]), v_values=np.array([1, 2, 3, 4, 5])
        )
    )


def test_wasserstein_distance_result_equal_false_different_y() -> None:
    assert not WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        WassersteinDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1])
        )
    )


def test_wasserstein_distance_result_equal_false_different_type() -> None:
    assert not WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(42)


def test_wasserstein_distance_result_equal_nan_true() -> None:
    assert WassersteinDistanceResult(
        u_values=np.array([1, 0, 0, 1, float("nan")]), v_values=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        WassersteinDistanceResult(
            u_values=np.array([1, 0, 0, 1, float("nan")]),
            v_values=np.array([1, 0, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_wasserstein_distance_result_equal_nan_false() -> None:
    assert not WassersteinDistanceResult(
        u_values=np.array([1, 0, 0, 1, float("nan")]), v_values=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        WassersteinDistanceResult(
            u_values=np.array([1, 0, 0, 1, float("nan")]),
            v_values=np.array([1, 0, 0, float("nan"), 1]),
        )
    )


def test_wasserstein_distance_result_compute_metrics_same() -> None:
    result = WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 5, "wasserstein_distance": 0.0},
    )


def test_wasserstein_distance_result_compute_metrics_different() -> None:
    result = WassersteinDistanceResult(u_values=np.array([0, 1, 3]), v_values=np.array([5, 6, 8]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 3, "wasserstein_distance": 5.0},
    )


def test_wasserstein_distance_result_compute_metrics_empty() -> None:
    result = WassersteinDistanceResult(u_values=np.array([]), v_values=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 0, "wasserstein_distance": float("nan")},
        equal_nan=True,
    )


def test_wasserstein_distance_result_compute_metrics_prefix_suffix() -> None:
    result = WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]),
        v_values=np.array([1, 2, 3, 4, 5]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_wasserstein_distance_suffix": 0.0},
    )


def test_wasserstein_distance_result_generate_figures() -> None:
    result = WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(result.generate_figures(), {})


def test_wasserstein_distance_result_generate_figures_empty() -> None:
    result = WassersteinDistanceResult(u_values=np.array([]), v_values=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_wasserstein_distance_result_generate_figures_prefix_suffix() -> None:
    result = WassersteinDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
