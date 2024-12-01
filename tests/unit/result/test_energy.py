from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import EnergyDistanceResult

##########################################
#     Tests for EnergyDistanceResult     #
##########################################


def test_energy_distance_result_u_values() -> None:
    assert objects_are_allclose(
        EnergyDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1])
        ).u_values,
        np.array([1, 2, 3, 4, 5]),
    )


def test_energy_distance_result_u_values_2d() -> None:
    assert objects_are_allclose(
        EnergyDistanceResult(
            u_values=np.array([[1, 2, 3], [4, 5, 6]]), v_values=np.array([[6, 5, 4], [3, 2, 1]])
        ).u_values,
        np.array([1, 2, 3, 4, 5, 6]),
    )


def test_energy_distance_result_v_values() -> None:
    assert objects_are_allclose(
        EnergyDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1])
        ).v_values,
        np.array([5, 4, 3, 2, 1]),
    )


def test_energy_distance_result_v_values_2d() -> None:
    assert objects_are_allclose(
        EnergyDistanceResult(
            u_values=np.array([[1, 2, 3], [4, 5, 6]]), v_values=np.array([[6, 5, 4], [3, 2, 1]])
        ).v_values,
        np.array([6, 5, 4, 3, 2, 1]),
    )


def test_energy_distance_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        EnergyDistanceResult(
            u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5, 0])
        )


def test_energy_distance_result_repr() -> None:
    assert repr(
        EnergyDistanceResult(u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5]))
    ).startswith("EnergyDistanceResult(")


def test_energy_distance_result_str() -> None:
    assert str(
        EnergyDistanceResult(u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5]))
    ).startswith("EnergyDistanceResult(")


def test_energy_distance_result_equal_true() -> None:
    assert EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        EnergyDistanceResult(u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5]))
    )


def test_energy_distance_result_equal_false_different_x() -> None:
    assert not EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        EnergyDistanceResult(u_values=np.array([1, 0, 0, 1, 0]), v_values=np.array([1, 2, 3, 4, 5]))
    )


def test_energy_distance_result_equal_false_different_y() -> None:
    assert not EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(
        EnergyDistanceResult(u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([5, 4, 3, 2, 1]))
    )


def test_energy_distance_result_equal_false_different_type() -> None:
    assert not EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    ).equal(42)


def test_energy_distance_result_equal_nan_true() -> None:
    assert EnergyDistanceResult(
        u_values=np.array([1, 0, 0, 1, float("nan")]), v_values=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        EnergyDistanceResult(
            u_values=np.array([1, 0, 0, 1, float("nan")]),
            v_values=np.array([1, 0, 0, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_energy_distance_result_equal_nan_false() -> None:
    assert not EnergyDistanceResult(
        u_values=np.array([1, 0, 0, 1, float("nan")]), v_values=np.array([1, 0, 0, float("nan"), 1])
    ).equal(
        EnergyDistanceResult(
            u_values=np.array([1, 0, 0, 1, float("nan")]),
            v_values=np.array([1, 0, 0, float("nan"), 1]),
        )
    )


def test_energy_distance_result_compute_metrics_same() -> None:
    result = EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 5, "energy_distance": 0.0},
    )


def test_energy_distance_result_compute_metrics_different() -> None:
    result = EnergyDistanceResult(
        u_values=np.array([0, 0, 0, 0, 0]), v_values=np.array([2, 2, 2, 2, 2])
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 5, "energy_distance": 2.0},
    )


def test_energy_distance_result_compute_metrics_empty() -> None:
    result = EnergyDistanceResult(u_values=np.array([]), v_values=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"count": 0, "energy_distance": float("nan")},
        equal_nan=True,
    )


def test_energy_distance_result_compute_metrics_prefix_suffix() -> None:
    result = EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]),
        v_values=np.array([1, 2, 3, 4, 5]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_count_suffix": 5, "prefix_energy_distance_suffix": 0.0},
    )


def test_energy_distance_result_generate_figures() -> None:
    result = EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(result.generate_figures(), {})


def test_energy_distance_result_generate_figures_empty() -> None:
    result = EnergyDistanceResult(u_values=np.array([]), v_values=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_energy_distance_result_generate_figures_prefix_suffix() -> None:
    result = EnergyDistanceResult(
        u_values=np.array([1, 2, 3, 4, 5]), v_values=np.array([1, 2, 3, 4, 5])
    )
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
