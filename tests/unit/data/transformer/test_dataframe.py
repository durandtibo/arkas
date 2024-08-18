from __future__ import annotations

import polars as pl
from coola import objects_are_equal
from grizz.transformer import Cast

from arkas.data.transformer import DataFrameTransformer

#########################################
#    Tests for DataFrameTransformer     #
#########################################


def test_dataframe_transformer_repr() -> None:
    assert repr(
        DataFrameTransformer(
            transformer=Cast(columns=None, dtype=pl.Int64), in_key="frame", out_key="frame"
        )
    ).startswith("DataFrameTransformer(")


def test_dataframe_transformer_str() -> None:
    assert str(
        DataFrameTransformer(
            transformer=Cast(columns=None, dtype=pl.Int64), in_key="frame", out_key="frame"
        )
    ).startswith("DataFrameTransformer(")


def test_dataframe_transformer_transform() -> None:
    data = {
        "frame": pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
            }
        )
    }
    out = DataFrameTransformer(
        transformer=Cast(columns=["col2"], dtype=pl.Int64), in_key="frame", out_key="frame"
    ).transform(data)
    assert out is not data
    assert objects_are_equal(
        out,
        {
            "frame": pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            )
        },
    )


def test_dataframe_transformer_transform_keys() -> None:
    data = {
        "in": pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
            }
        )
    }
    out = DataFrameTransformer(
        transformer=Cast(columns=["col2"], dtype=pl.Int64), in_key="in", out_key="out"
    ).transform(data)
    assert out is not data
    assert objects_are_equal(
        out,
        {
            "in": pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            "out": pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": [1, 2, 3, 4, 5],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
        },
    )
