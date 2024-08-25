from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric.utils import multi_isnan

#################################
#     Tests for multi_isnan     #
#################################


def test_multi_isnan_1_array() -> None:
    assert objects_are_equal(
        multi_isnan([np.array([1, 0, 0, 1, float("nan")])]),
        np.array([False, False, False, False, True]),
    )


def test_multi_isnan_2_arrays() -> None:
    assert objects_are_equal(
        multi_isnan(
            [
                np.array([1, 0, 0, 1, float("nan"), float("nan")]),
                np.array([1, float("nan"), 0, 1, 1, float("nan")]),
            ]
        ),
        np.array([False, True, False, False, True, True]),
    )


def test_multi_isnan_3_arrays() -> None:
    assert objects_are_equal(
        multi_isnan(
            [
                np.array([1, 0, 0, 1, float("nan"), float("nan")]),
                np.array([1, float("nan"), 0, 1, 1, float("nan")]),
                np.array([float("nan"), 1, 0, 1, 1, float("nan")]),
            ]
        ),
        np.array([True, True, False, False, True, True]),
    )


def test_multi_isnan_empty() -> None:
    with pytest.raises(RuntimeError, match="'arrays' cannot be empty"):
        multi_isnan([])
