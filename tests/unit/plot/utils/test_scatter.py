from __future__ import annotations

import pytest
from coola import objects_are_allclose

from arkas.plot.utils.scatter import find_alpha_from_size, find_marker_size_from_size

##########################################
#     Tests for find_alpha_from_size     #
##########################################


@pytest.mark.parametrize(
    ("n", "alpha"),
    [
        (100, 1.0),
        (1_000, 1.0),
        (50_500, 0.6),
        (100_000, 0.2),
        (1_000_000, 0.2),
    ],
)
def test_find_alpha_from_size_default(n: int, alpha: float) -> None:
    assert objects_are_allclose(find_alpha_from_size(n), alpha)


@pytest.mark.parametrize(
    ("n", "alpha"),
    [
        (10, 0.9),
        (100, 0.9),
        (50_050.0, 0.5),
        (100_000, 0.1),
        (1_000_000, 0.1),
    ],
)
def test_find_alpha_from_size(n: int, alpha: float) -> None:
    assert objects_are_allclose(
        find_alpha_from_size(n, min_alpha=(0.1, 100_000), max_alpha=(0.9, 100)), alpha
    )


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
