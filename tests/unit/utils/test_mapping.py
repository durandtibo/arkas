from __future__ import annotations

from arkas.utils.mapping import find_missing_keys

#######################################
#     Tests for find_missing_keys     #
#######################################


def test_find_missing_keys_all_present() -> None:
    assert (
        find_missing_keys(mapping={"key1": 1, "key2": 2, "key3": 3}, keys=["key1", "key2"]) == set()
    )


def test_find_missing_keys_all_missing() -> None:
    assert find_missing_keys(mapping={}, keys=["key1", "key2"]) == {"key1", "key2"}


def test_find_missing_keys_partially_present() -> None:
    assert find_missing_keys(
        mapping={"key1": 1, "key2": 2, "key3": 3}, keys=["key1", "key2", "key4"]
    ) == {"key4"}


def test_find_missing_keys_empty() -> None:
    assert find_missing_keys(mapping={}, keys=[]) == set()


def test_find_missing_keys_empty_mapping() -> None:
    assert find_missing_keys(mapping={}, keys=["key1", "key2"]) == {"key1", "key2"}


def test_find_missing_keys_empty_keys() -> None:
    assert find_missing_keys(mapping={"key1": 1, "key2": 2}, keys=[]) == set()
