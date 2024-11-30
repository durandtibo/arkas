from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_allclose

from arkas.metric import spearmanr
from arkas.testing import scipy_available

###############################
#     Tests for spearmanr     #
###############################


@scipy_available
def test_spearmanr_perfect_positive_correlation() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_perfect_positive_correlation_2d() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[1, 2, 3], [4, 5, 6]])),
        {"count": 6, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_perfect_negative_correlation() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([4, 3, 2, 1]), y=np.array([1, 2, 3, 4])),
        {"count": 4, "spearman_coeff": -1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_perfect_no_correlation() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([-2, -1, 0, 1, 2]), y=np.array([0, 1, -1, 1, 0])),
        {"count": 5, "spearman_coeff": 0.0, "spearman_pvalue": 1.0},
    )


@scipy_available
def test_spearmanr_constant() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([1, 1, 1, 1, 1]), y=np.array([1, 1, 1, 1, 1])),
        {"count": 5, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearmanr_empty() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([]), y=np.array([])),
        {"count": 0, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearmanr_alternative_less() -> None:
    assert objects_are_allclose(
        spearmanr(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]), alternative="less"),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 1.0},
    )


@scipy_available
def test_spearmanr_alternative_greater() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            alternative="greater",
        ),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_prefix_suffix() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
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
def test_spearmanr_nan_omit() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 4, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_omit_x() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([1, 2, 3, 4, 5, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_omit_y() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([1, 2, 3, 4, 5, 0]),
            y=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "spearman_coeff": 1.0, "spearman_pvalue": 0.0},
    )


@scipy_available
def test_spearmanr_nan_propagate() -> None:
    assert objects_are_allclose(
        spearmanr(
            x=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        ),
        {"count": 7, "spearman_coeff": float("nan"), "spearman_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_spearmanr_nan_raise() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        spearmanr(
            x=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


@scipy_available
def test_spearmanr_nan_raise_x() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        spearmanr(
            x=np.array([float("nan"), 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            nan_policy="raise",
        )


@scipy_available
def test_spearmanr_nan_raise_y() -> None:
    with pytest.raises(ValueError, match="'y' contains at least one NaN value"):
        spearmanr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, float("nan")]),
            nan_policy="raise",
        )
