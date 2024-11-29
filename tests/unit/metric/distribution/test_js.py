from __future__ import annotations

import numpy as np
from coola import objects_are_allclose

from arkas.metric import js_div
from arkas.testing import scipy_available

############################
#     Tests for js_div     #
############################


@scipy_available
def test_js_div_same() -> None:
    assert objects_are_allclose(
        js_div(p=np.array([0.1, 0.6, 0.1, 0.2]), q=np.array([0.1, 0.6, 0.1, 0.2])),
        {"size": 4, "js_div": 0.0},
    )


@scipy_available
def test_js_div_different() -> None:
    assert objects_are_allclose(
        js_div(p=np.array([0.10, 0.40, 0.50]), q=np.array([0.80, 0.15, 0.05])),
        {"size": 3, "js_div": 0.29126084062606405},
    )


@scipy_available
def test_js_div_empty() -> None:
    assert objects_are_allclose(
        js_div(p=np.array([]), q=np.array([])),
        {"size": 0, "js_div": float("nan")},
        equal_nan=True,
    )


@scipy_available
def test_js_div_prefix_suffix() -> None:
    assert objects_are_allclose(
        js_div(
            p=np.array([0.1, 0.6, 0.1, 0.2]),
            q=np.array([0.1, 0.6, 0.1, 0.2]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_size_suffix": 4, "prefix_js_div_suffix": 0.0},
    )
