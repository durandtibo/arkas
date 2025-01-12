from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from polars.testing import assert_frame_equal

from arkas.utils.dataframe import check_column_exist, check_num_columns, to_arrays

########################################
#     Tests for check_column_exist     #
########################################


def test_check_column_exist() -> None:
    check_column_exist(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        ),
        "col1",
    )


def test_check_column_exist_missing() -> None:
    with pytest.raises(ValueError, match="The column 'col' is not in the DataFrame:"):
        check_column_exist(
            pl.DataFrame(
                {
                    "col1": [0, 1, 1, 0, 0, 1, 0],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            ),
            "col",
        )


#######################################
#     Tests for check_num_columns     #
#######################################


def test_check_num_columns() -> None:
    check_num_columns(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        ),
        num_columns=3,
    )


def test_check_num_columns_incorrect() -> None:
    with pytest.raises(
        ValueError, match="The DataFrame must have 2 columns but received a DataFrame"
    ):
        check_num_columns(
            pl.DataFrame(
                {
                    "col1": [0, 1, 1, 0, 0, 1, 0],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            ),
            num_columns=2,
        )


###############################
#     Tests for to_arrays     #
###############################


def test_to_arrays_empty() -> None:
    frame = pl.DataFrame({})
    assert_frame_equal(frame, pl.DataFrame())
    assert objects_are_equal(to_arrays(frame), {})


def test_to_arrays_1_col() -> None:
    data = {"int": np.array([1, 2, 3, 4, 5], dtype=np.int64)}
    frame = pl.DataFrame(data)
    assert_frame_equal(
        frame,
        pl.DataFrame({"int": [1, 2, 3, 4, 5]}, schema={"int": pl.Int64}),
    )
    assert objects_are_equal(to_arrays(frame), data)


def test_to_arrays_multiple_cols() -> None:
    data = {
        "int": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64),
        "str": np.array(["a", "b", "c", "d", "e"], dtype=np.object_),
    }
    frame = pl.DataFrame(data)
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {
                "int": [1, 2, 3, 4, 5],
                "float": [5.0, 4.0, 3.0, 2.0, 1.0],
                "str": ["a", "b", "c", "d", "e"],
            },
            schema={"int": pl.Int64, "float": pl.Float64, "str": pl.String},
        ),
    )
    assert objects_are_equal(to_arrays(frame), data)


def test_to_arrays_dims() -> None:
    data = {
        "1d": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "2d": np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]], dtype=np.int64),
    }
    frame = pl.DataFrame(data)
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"1d": [1, 2, 3, 4, 5], "2d": [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]]},
            schema={"1d": pl.Int64, "2d": pl.Array(pl.Int64, 2)},
        ),
    )
    assert objects_are_equal(to_arrays(frame), data)


def test_to_arrays_arrays() -> None:
    data = {"int": np.array([1, 2, 3, 4, 5]), "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0])}
    frame = pl.DataFrame(data)
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"int": [1, 2, 3, 4, 5], "float": [5.0, 4.0, 3.0, 2.0, 1.0]},
            schema={"int": pl.Int64, "float": pl.Float64},
        ),
    )
    assert objects_are_equal(to_arrays(frame), data)


def test_to_arrays_sequence() -> None:
    frame = pl.DataFrame({"int": [1, 2, 3, 4, 5], "float": (5.0, 4.0, 3.0, 2.0, 1.0)})
    assert_frame_equal(
        frame,
        pl.DataFrame(
            {"int": [1, 2, 3, 4, 5], "float": [5.0, 4.0, 3.0, 2.0, 1.0]},
            schema={"int": pl.Int64, "float": pl.Float64},
        ),
    )
    assert objects_are_equal(
        to_arrays(frame),
        {
            "int": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "float": np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64),
        },
    )
