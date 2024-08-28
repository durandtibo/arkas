from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from arkas.utils.array import to_array

if TYPE_CHECKING:
    from collections.abc import Sequence


##############################
#     Tests for to_array     #
##############################


@pytest.mark.parametrize(
    "data",
    [
        np.array([3, 1, 2, 0, 1]),
        [3, 1, 2, 0, 1],
        (3, 1, 2, 0, 1),
        pl.Series([3, 1, 2, 0, 1]),
    ],
)
def test_to_array_int(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3, 1, 2, 0, 1], dtype=int))


@pytest.mark.parametrize(
    "data",
    [
        np.array([3.0, 1.0, 2.0, 0.0, 1.0]),
        [3.0, 1.0, 2.0, 0.0, 1.0],
        (3.0, 1.0, 2.0, 0.0, 1.0),
        pl.Series([3.0, 1.0, 2.0, 0.0, 1.0]),
    ],
)
def test_to_array_float(data: Sequence | np.ndarray) -> None:
    assert np.array_equal(to_array(data), np.array([3.0, 1.0, 2.0, 0.0, 1.0], dtype=float))