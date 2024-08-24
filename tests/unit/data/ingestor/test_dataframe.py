from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.ingestor import BaseIngestor, Ingestor

from arkas.data.ingestor import DataFrameIngestor


@pytest.fixture
def ingestor() -> BaseIngestor:
    return Ingestor(
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
            }
        )
    )


######################################
#    Tests for DataFrameIngestor     #
######################################


def test_dataframe_ingestor_repr(ingestor: BaseIngestor) -> None:
    assert repr(DataFrameIngestor(ingestor=ingestor)).startswith("DataFrameIngestor(")


def test_dataframe_ingestor_str(ingestor: BaseIngestor) -> None:
    assert str(DataFrameIngestor(ingestor=ingestor)).startswith("DataFrameIngestor(")


def test_dataframe_ingestor_ingest(ingestor: BaseIngestor) -> None:
    data = DataFrameIngestor(ingestor=ingestor).ingest()
    assert objects_are_equal(
        data,
        pl.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": ["1", "2", "3", "4", "5"],
                "col3": ["a", "b", "c", "d", "e"],
            }
        ),
    )
