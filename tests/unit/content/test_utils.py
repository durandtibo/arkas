from __future__ import annotations

from typing import Any

import pytest

from arkas.content.utils import float_to_str, to_str

############################
#     Tests for to_str     #
############################


@pytest.mark.parametrize(
    ("value", "string"),
    [
        ("meow", "meow"),
        (42, "42"),
        (42.1, "42.1"),
        (None, "None"),
    ],
)
def test_to_str(value: Any, string: str) -> None:
    assert to_str(value) == string


##################################
#     Tests for float_to_str     #
##################################


@pytest.mark.parametrize(
    ("value", "string"),
    [
        (0, "0"),
        (42.0, "42"),
        (42.1, "42.1"),
        (1e9, "1e+09"),
        (-1e9, "-1e+09"),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
    ],
)
def test_float_to_str(value: float, string: str) -> None:
    assert float_to_str(value) == string
