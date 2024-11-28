from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.metric import pearsonr
from arkas.testing import scipy_available

##############################
#     Tests for pearsonr     #
##############################


@scipy_available
def test_pearsonr_perfect_positive_correlation() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_positive_correlation_2d() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])),
        {"count": 6, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_negative_correlation() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "pearson_coeff": -1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_no_correlation() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([-2, -1, 0, 1, 2]), y_pred=np.array([0, 1, -1, 1, 0])),
        {"count": 5, "pearson_coeff": 0.0, "pearson_pvalue": 1.0},
    )


@scipy_available
def test_pearsonr_constant() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([1, 1, 1, 1, 1]), y_pred=np.array([1, 1, 1, 1, 1])),
        {"count": 5, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_empty() -> None:
    assert objects_are_allclose(
        pearsonr(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_alternative_less() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]), alternative="less"
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 1.0},
    )


@scipy_available
def test_pearsonr_alternative_greater() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            alternative="greater",
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_prefix_suffix() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {
            "prefix_count_suffix": 5,
            "prefix_pearson_coeff_suffix": 1.0,
            "prefix_pearson_pvalue_suffix": 0.0,
        },
    )


@scipy_available
def test_pearsonr_nan() -> None:
    with pytest.raises(ValueError, match="array must not contain infs or NaNs"):
        pearsonr(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
        )


@scipy_available
def test_pearsonr_drop_nan() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
            drop_nan=True,
        ),
        {"count": 4, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_drop_nan_y_true() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            drop_nan=True,
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_drop_nan_y_pred() -> None:
    assert objects_are_allclose(
        pearsonr(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            drop_nan=True,
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )
