from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.data.ingestor import Ingestor


@pytest.fixture
def data() -> dict:
    return {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([0, 1, 2, 3, 2])}


##############################
#     Tests for Ingestor     #
##############################


def test_ingestor_repr(data: dict) -> None:
    assert repr(Ingestor(data=data)).startswith("Ingestor(")


def test_ingestor_str(data: dict) -> None:
    assert str(Ingestor(data=data)).startswith("Ingestor(")


def test_ingestor_ingest(data: dict) -> None:
    out = Ingestor(data=data).ingest()
    assert data is not out
    assert objects_are_equal(
        out, {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([0, 1, 2, 3, 2])}
    )
