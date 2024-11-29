from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import KLDivResult

#################################
#     Tests for KLDivResult     #
#################################


def test_kl_div_result_p() -> None:
    assert objects_are_allclose(
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_kl_div_result_p_2d() -> None:
    assert objects_are_allclose(
        KLDivResult(p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_kl_div_result_q() -> None:
    assert objects_are_allclose(
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_kl_div_result_q_2d() -> None:
    assert objects_are_allclose(
        KLDivResult(p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_kl_div_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2, 0]))


def test_kl_div_result_repr() -> None:
    assert repr(
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    ).startswith("KLDivResult(")


def test_kl_div_result_str() -> None:
    assert str(
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    ).startswith("KLDivResult(")


def test_kl_div_result_equal_true() -> None:
    assert KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])).equal(
        KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    )


def test_kl_div_result_equal_false_different_p() -> None:
    assert not KLDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(KLDivResult(p=np.array([0.1, 0.6, 0.2, 0.1]), q=np.array([0.1, 0.6, 0.1, 0.2])))


def test_kl_div_result_equal_false_different_q() -> None:
    assert not KLDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])))


def test_kl_div_result_equal_false_different_type() -> None:
    assert not KLDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(42)


def test_kl_div_result_equal_nan_true() -> None:
    assert KLDivResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        KLDivResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        ),
        equal_nan=True,
    )


def test_kl_div_result_equal_nan_false() -> None:
    assert not KLDivResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        KLDivResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        )
    )


def test_kl_div_result_compute_metrics_same() -> None:
    result = KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 4, "kl_pq": 0.0, "kl_qp": 0.0},
    )


def test_kl_div_result_compute_metrics_different() -> None:
    result = KLDivResult(p=np.array([0.10, 0.40, 0.50]), q=np.array([0.80, 0.15, 0.05]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 3, "kl_pq": 1.3356800935337299, "kl_qp": 1.4012995907424075},
    )


def test_kl_div_result_compute_metrics_empty() -> None:
    result = KLDivResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 0, "kl_pq": float("nan"), "kl_qp": float("nan")},
        equal_nan=True,
    )


def test_kl_div_result_compute_metrics_prefix_suffix() -> None:
    result = KLDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]),
        q=np.array([0.1, 0.6, 0.1, 0.2]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_size_suffix": 4, "prefix_kl_pq_suffix": 0.0, "prefix_kl_qp_suffix": 0.0},
    )


def test_kl_div_result_generate_figures() -> None:
    result = KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_kl_div_result_generate_figures_empty() -> None:
    result = KLDivResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_kl_div_result_generate_figures_prefix_suffix() -> None:
    result = KLDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
