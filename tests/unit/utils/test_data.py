from __future__ import annotations

import polars as pl

from arkas.utils.data import find_keys

###############################
#     Tests for find_keys     #
###############################


def test_find_keys_dict() -> None:
    assert find_keys({"pred": None, "target": None}) == {"pred", "target"}


def test_find_keys_dict_empty() -> None:
    assert find_keys({}) == set()


def test_find_keys_dataframe() -> None:
    assert find_keys(pl.DataFrame({"pred": [], "target": []})) == {"pred", "target"}


def test_find_keys_dataframe_empty() -> None:
    assert find_keys(pl.DataFrame()) == set()
