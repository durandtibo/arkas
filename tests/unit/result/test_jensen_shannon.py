from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.result import JensenShannonDivergenceResult

###################################################
#     Tests for JensenShannonDivergenceResult     #
###################################################


def test_jensen_shannon_divergence_result_p() -> None:
    assert objects_are_allclose(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])
        ).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_jensen_shannon_divergence_result_p_2d() -> None:
    assert objects_are_allclose(
        JensenShannonDivergenceResult(
            p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])
        ).p,
        np.array([0.1, 0.6, 0.1, 0.2]),
    )


def test_jensen_shannon_divergence_result_q() -> None:
    assert objects_are_allclose(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])
        ).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_jensen_shannon_divergence_result_q_2d() -> None:
    assert objects_are_allclose(
        JensenShannonDivergenceResult(
            p=np.array([[0.1, 0.6], [0.1, 0.2]]), q=np.array([[0.2, 0.5], [0.2, 0.1]])
        ).q,
        np.array([0.2, 0.5, 0.2, 0.1]),
    )


def test_jensen_shannon_divergence_result_y_incorrect_shape() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2, 0])
        )


def test_jensen_shannon_divergence_result_repr() -> None:
    assert repr(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
        )
    ).startswith("JensenShannonDivergenceResult(")


def test_jensen_shannon_divergence_result_str() -> None:
    assert str(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
        )
    ).startswith("JensenShannonDivergenceResult(")


def test_jensen_shannon_divergence_result_equal_true() -> None:
    assert JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
        )
    )


def test_jensen_shannon_divergence_result_equal_false_different_p() -> None:
    assert not JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.2, 0.1]), q=np.array([0.1, 0.6, 0.1, 0.2])
        )
    )


def test_jensen_shannon_divergence_result_equal_false_different_q() -> None:
    assert not JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(
        JensenShannonDivergenceResult(
            p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])
        )
    )


def test_jensen_shannon_divergence_result_equal_false_different_type() -> None:
    assert not JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    ).equal(42)


def test_jensen_shannon_divergence_result_equal_nan_true() -> None:
    assert JensenShannonDivergenceResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        JensenShannonDivergenceResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        ),
        equal_nan=True,
    )


def test_jensen_shannon_divergence_result_equal_nan_false() -> None:
    assert not JensenShannonDivergenceResult(
        p=np.array([1, 0, 0, 0, float("nan")]), q=np.array([1, 0, 0, float("nan"), 0])
    ).equal(
        JensenShannonDivergenceResult(
            p=np.array([1, 0, 0, 0, float("nan")]),
            q=np.array([1, 0, 0, float("nan"), 0]),
        )
    )


def test_jensen_shannon_divergence_result_compute_metrics_same() -> None:
    result = JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 4, "jensen_shannon_divergence": 0.0},
    )


def test_jensen_shannon_divergence_result_compute_metrics_different() -> None:
    result = JensenShannonDivergenceResult(
        p=np.array([0.10, 0.40, 0.50]), q=np.array([0.80, 0.15, 0.05])
    )
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 3, "jensen_shannon_divergence": 0.29126084062606405},
    )


def test_jensen_shannon_divergence_result_compute_metrics_empty() -> None:
    result = JensenShannonDivergenceResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(
        result.compute_metrics(),
        {"size": 0, "jensen_shannon_divergence": float("nan")},
        equal_nan=True,
    )


def test_jensen_shannon_divergence_result_compute_metrics_prefix_suffix() -> None:
    result = JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]),
        q=np.array([0.1, 0.6, 0.1, 0.2]),
    )
    assert objects_are_allclose(
        result.compute_metrics(prefix="prefix_", suffix="_suffix"),
        {"prefix_size_suffix": 4, "prefix_jensen_shannon_divergence_suffix": 0.0},
    )


def test_jensen_shannon_divergence_result_generate_figures() -> None:
    result = JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    )
    assert objects_are_allclose(result.generate_figures(), {})


def test_jensen_shannon_divergence_result_generate_figures_empty() -> None:
    result = JensenShannonDivergenceResult(p=np.array([]), q=np.array([]))
    assert objects_are_allclose(result.generate_figures(), {})


def test_jensen_shannon_divergence_result_generate_figures_prefix_suffix() -> None:
    result = JensenShannonDivergenceResult(
        p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])
    )
    assert objects_are_allclose(result.generate_figures(prefix="prefix_", suffix="_suffix"), {})
