from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from arkas.utils.array import to_array

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
def test_to_array_int(data: Any) -> None:
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
def test_to_array_float(data: Any) -> None:
    assert np.array_equal(to_array(data), np.array([3.0, 1.0, 2.0, 0.0, 1.0], dtype=float))


def test_to_array_dataframe_1_col() -> None:
    assert np.array_equal(
        to_array(pl.DataFrame({"col": [3, 1, 2, 0, 1]})),
        np.array([[3], [1], [2], [0], [1]]),
    )


def test_to_array_dataframe_2_cols() -> None:
    assert np.array_equal(
        to_array(pl.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [0, 1, 0, 1, 0]})),
        np.array([[1, 0], [2, 1], [3, 0], [4, 1], [5, 0]]),
    )
