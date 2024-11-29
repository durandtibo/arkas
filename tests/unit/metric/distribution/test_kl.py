from __future__ import annotations

import numpy as np
from coola import objects_are_allclose

from arkas.metric import kl_div
from arkas.testing import scipy_available

############################
#     Tests for kl_div     #
############################


@scipy_available
def test_kl_div_same() -> None:
    assert objects_are_allclose(
        kl_div(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])),
        {"size": 4, "kl_pq": 0.0, "kl_qp": 0.0},
    )


@scipy_available
def test_kl_div_different() -> None:
    assert objects_are_allclose(
        kl_div(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.2, 0.5, 0.2, 0.1])),
        {"size": 4, "kl_pq": 0.10939293407637275, "kl_qp": 0.1167833757710063},
    )


@scipy_available
def test_kl_div_empty() -> None:
    assert objects_are_allclose(
        kl_div(p=np.array([]), q=np.array([])),
        {"size": 0, "kl_pq": float("nan"), "kl_qp": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_kl_div_prefix_suffix() -> None:
    assert objects_are_allclose(
        kl_div(
            p=np.array([0.1, 0.6, 0.1, 0.2]),
            q=np.array([0.1, 0.6, 0.1, 0.2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_size_suffix": 4, "prefix_kl_pq_suffix": 0.0, "prefix_kl_qp_suffix": 0.0},
    )
