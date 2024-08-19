from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

from arkas.data.transformer import ColumnToArrayTransformer

#############################################
#    Tests for ColumnToArrayTransformer     #
#############################################


def test_dataframe_transformer_repr() -> None:
    assert repr(ColumnToArrayTransformer(col="col1", in_key="frame", out_key="label")).startswith(
        "ColumnToArrayTransformer("
    )


def test_dataframe_transformer_str() -> None:
    assert str(ColumnToArrayTransformer(col="col1", in_key="frame", out_key="label")).startswith(
        "ColumnToArrayTransformer("
    )


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
    out = ColumnToArrayTransformer(col="col1", in_key="frame", out_key="label").transform(data)
    assert out is not data
    assert objects_are_equal(
        out,
        {
            "frame": pl.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5],
                    "col2": ["1", "2", "3", "4", "5"],
                    "col3": ["a", "b", "c", "d", "e"],
                }
            ),
            "label": np.array([1, 2, 3, 4, 5]),
        },
    )
