from __future__ import annotations

import numpy as np
from coola import objects_are_allclose

from arkas.metric import spearman_correlation
from arkas.testing import scipy_available

##########################################
#     Tests for spearman_correlation     #
##########################################


@scipy_available
def test_spearman_correlation_perfect_positive_correlation() -> None:
    assert objects_are_allclose(
        spearman_correlation(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearman_correlation_perfect_positive_correlation_2d() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearman_correlation_perfect_negative_correlation() -> None:
    assert objects_are_allclose(
        spearman_correlation(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "spearman_coeff": -1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearman_correlation_perfect_no_correlation() -> None:
    assert objects_are_allclose(
        spearman_correlation(y_true=np.array([-2, -1, 0, 1, 2]), y_pred=np.array([0, 1, -1, 1, 0])),
        {"count": 5, "spearman_coeff": 0.0, "spearman_pvalue": 1.0},
    )


@scipy_available
def test_spearman_correlation_constant() -> None:
    assert objects_are_allclose(
        spearman_correlation(y_true=np.array([1, 1, 1, 1, 1]), y_pred=np.array([1, 1, 1, 1, 1])),
        {"count": 5, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearman_correlation_empty() -> None:
    assert objects_are_allclose(
        spearman_correlation(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearman_correlation_prefix_suffix() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_spearman_coeff_suffix": 1.0,
            "prefix_spearman_pvalue_suffix": 0.0,
        },
    )


@scipy_available
def test_spearman_correlation_nan() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
        ),
        {"count": 6, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearman_correlation_ignore_nan() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
            ignore_nan=True,
        ),
        {"count": 4, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearman_correlation_ignore_nan_y_true() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            ignore_nan=True,
        ),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearman_correlation_ignore_nan_y_pred() -> None:
    assert objects_are_allclose(
        spearman_correlation(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            ignore_nan=True,
        ),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )
