from __future__ import annotations

import polars as pl

from arkas.utils.data import find_keys, find_missing_keys, flat_keys

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


#######################################
#     Tests for find_missing_keys     #
#######################################


def test_find_missing_keys_all_present() -> None:
    assert find_missing_keys(keys={"key1", "key2", "key3"}, queries=["key1", "key2"]) == set()


def test_find_missing_keys_all_missing() -> None:
    assert find_missing_keys(keys=[], queries=["key1", "key2"]) == {"key1", "key2"}


def test_find_missing_keys_partially_present() -> None:
    assert find_missing_keys(keys={"key1", "key2", "key3"}, queries=["key1", "key2", "key4"]) == {
        "key4"
    }


def test_find_missing_keys_empty() -> None:
    assert find_missing_keys(keys=[], queries=[]) == set()


def test_find_missing_keys_empty_keys() -> None:
    assert find_missing_keys(keys={}, queries=["key1", "key2"]) == {"key1", "key2"}


def test_find_missing_keys_empty_queries() -> None:
    assert find_missing_keys(keys={"key1", "key2"}, queries=[]) == set()


###############################
#     Tests for flat_keys     #
###############################


def test_flat_keys() -> None:
    assert flat_keys(["key0", "key1", "key2"]) == ["key0", "key1", "key2"]


def test_flat_keys_mixed() -> None:
    assert flat_keys(["key0", ["key1"], ["key2", "key3", "key4"]]) == [
        "key0",
        "key1",
        "key2",
        "key3",
        "key4",
    ]


def test_flat_keys_nested() -> None:
    assert flat_keys([["key0"], ["key1"], ["key2", "key3", "key4"]]) == [
        "key0",
        "key1",
        "key2",
        "key3",
        "key4",
    ]


def test_flat_keys_empty() -> None:
    assert flat_keys([]) == []
