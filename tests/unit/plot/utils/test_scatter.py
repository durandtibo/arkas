from __future__ import annotations

import pytest
from coola import objects_are_allclose

from arkas.plot.utils.scatter import find_marker_size_from_size

################################################
#     Tests for find_marker_size_from_size     #
################################################


@pytest.mark.parametrize(
    ("n", "marker_size"),
    [
        (100, 32.0),
        (1_000, 32.0),
        (50_500, 21.0),
        (100_000, 10.0),
        (1_000_000, 10.0),
    ],
)
def test_find_marker_size_from_size_default(n: int, marker_size: float) -> None:
    assert objects_are_allclose(find_marker_size_from_size(n), marker_size)


@pytest.mark.parametrize(
    ("n", "marker_size"),
    [
        (10, 50.0),
        (100, 50.0),
        (50_050.0, 27.5),
        (100_000, 5.0),
        (1_000_000, 5.0),
    ],
)
def test_find_marker_size_from_size(n: int, marker_size: float) -> None:
    assert objects_are_allclose(
        find_marker_size_from_size(n, min_size=(5.0, 100_000), max_size=(50.0, 100)), marker_size
    )
