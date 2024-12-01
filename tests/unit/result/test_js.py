from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import JSDivResult

#################################
#     Tests for JSDivResult     #
#################################


def test_js_div_result_p() -> None:
    assert objects_are_allclose(
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_js_div_result_p_2d() -> None:
    assert objects_are_allclose(
        JSDivResult(p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_js_div_result_q() -> None:
    assert objects_are_allclose(
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_js_div_result_q_2d() -> None:
    assert objects_are_allclose(
        JSDivResult(p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_js_div_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2, 0]))


def test_js_div_result_repr() -> None:
    assert repr(
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    ).startswith("JSDivResult(")


def test_js_div_result_str() -> None:
    assert str(
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    ).startswith("JSDivResult(")


def test_js_div_result_equal_true() -> None:
    assert JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])).equal(
        JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    )


def test_js_div_result_equal_false_different_p() -> None:
    assert not JSDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(JSDivResult(p=np.array([0.1, 0.6, 0.2, 0.1]), q=np.array([0.1, 0.6, 0.1, 0.2])))


def test_js_div_result_equal_false_different_q() -> None:
    assert not JSDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])))


def test_js_div_result_equal_false_different_type() -> None:
    assert not JSDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(42)


def test_js_div_result_equal_nan_true() -> None:
    assert JSDivResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        JSDivResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        ),
        equal_nan=True,
    )


def test_js_div_result_equal_nan_false() -> None:
    assert not JSDivResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        JSDivResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        )
    )


def test_js_div_result_compute_metrics_same() -> None:
    result = JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 4, "js_div": 0.0},
    )


def test_js_div_result_compute_metrics_different() -> None:
    result = JSDivResult(p=np.array([0.10, 0.40, 0.50]), q=np.array([0.80, 0.15, 0.05]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 3, "js_div": 0.29126084062606405},
    )


def test_js_div_result_compute_metrics_empty() -> None:
    result = JSDivResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 0, "js_div": float("nan")},
        equal_nan=True,
    )


def test_js_div_result_compute_metrics_prefix_suffix() -> None:
    result = JSDivResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]),
        q=np.array([0.1, 0.6, 0.1, 0.2]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_size_suffix": 4, "prefix_js_div_suffix": 0.0},
    )


def test_js_div_result_generate_figures() -> None:
    result = JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_js_div_result_generate_figures_empty() -> None:
    result = JSDivResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_js_div_result_generate_figures_prefix_suffix() -> None:
    result = JSDivResult(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2]))
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
