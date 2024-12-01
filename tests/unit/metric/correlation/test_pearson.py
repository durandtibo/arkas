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
        pearsonr(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_positive_correlation_2d() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([[1, 2, 3], [4, 5, 6]]), y=np.array([[1, 2, 3], [4, 5, 6]])),
        {"count": 6, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_negative_correlation() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([4, 3, 2, 1]), y=np.array([1, 2, 3, 4])),
        {"count": 4, "pearson_coeff": -1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_perfect_no_correlation() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([-2, -1, 0, 1, 2]), y=np.array([0, 1, -1, 1, 0])),
        {"count": 5, "pearson_coeff": 0.0, "pearson_pvalue": 1.0},
    )


@scipy_available
@pytest.mark.filterwarnings(
    "ignore:An input array is constant; the correlation coefficient is not defined."
)
def test_pearsonr_constant() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([1, 1, 1, 1, 1]), y=np.array([1, 1, 1, 1, 1])),
        {"count": 5, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_empty() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([]), y=np.array([])),
        {"count": 0, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_alternative_less() -> None:
    assert objects_are_allclose(
        pearsonr(x=np.array([1, 2, 3, 4, 5]), y=np.array([1, 2, 3, 4, 5]), alternative="less"),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 1.0},
    )


@scipy_available
def test_pearsonr_alternative_greater() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            alternative="greater",
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_prefix_suffix() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
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
def test_pearsonr_nan_omit() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 4, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_omit_x() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_omit_y() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5, 0]),
            y=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "pearson_coeff": 1.0, "pearson_pvalue": 0.0},
    )


@scipy_available
def test_pearsonr_nan_propagate() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([float("nan"), 2, 3, 4, 5, 6, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, float("nan"), float("nan")]),
        ),
        {"count": 7, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_propagate_x() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5, float("nan")]),
            y=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {"count": 6, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_propagate_y() -> None:
    assert objects_are_allclose(
        pearsonr(
            x=np.array([1, 2, 3, 4, 5, 0]),
            y=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "pearson_coeff": float("nan"), "pearson_pvalue": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_pearsonr_nan_raise() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        pearsonr(
            x=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


@scipy_available
def test_pearsonr_nan_raise_x() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        pearsonr(
            x=np.array([float("nan"), 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, 5]),
            nan_policy="raise",
        )


@scipy_available
def test_pearsonr_nan_raise_y() -> None:
    with pytest.raises(ValueError, match="'y' contains at least one NaN value"):
        pearsonr(
            x=np.array([1, 2, 3, 4, 5]),
            y=np.array([1, 2, 3, 4, float("nan")]),
            nan_policy="raise",
        )
