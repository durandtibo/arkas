from __future__ import annotations

import numpy as np
import polars as pl
from coola import objects_are_equal

from arkas.utils.data import find_keys, find_missing_keys, flat_keys, prepare_array

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


###################################
#     Tests for prepare_array     #
###################################


def test_prepare_array_dict_1_key_array() -> None:
    assert objects_are_equal(
        prepare_array({"key": np.array([1, 2, 3, 4, 5])}, keys="key"), np.array([1, 2, 3, 4, 5])
    )


def test_prepare_array_dict_1_key_list() -> None:
    assert objects_are_equal(
        prepare_array({"key": [1, 2, 3, 4, 5]}, keys="key"), np.array([1, 2, 3, 4, 5])
    )


def test_prepare_array_dict_2_keys_array() -> None:
    assert objects_are_equal(
        prepare_array(
            {"key1": np.array([1, 2, 3, 4, 5]), "key2": np.array([1, 0, 1, 0, 1])},
            keys=["key1", "key2"],
        ),
        np.array([[1, 1], [2, 0], [3, 1], [4, 0], [5, 1]]),
    )


def test_prepare_array_dict_2_keys_list() -> None:
    assert objects_are_equal(
        prepare_array(
            {"key1": [1, 2, 3, 4, 5], "key2": [1, 0, 1, 0, 1]},
            keys=["key1", "key2"],
        ),
        np.array([[1, 1], [2, 0], [3, 1], [4, 0], [5, 1]]),
    )


def test_prepare_array_dataframe_1_col() -> None:
    assert objects_are_equal(
        prepare_array(pl.DataFrame({"key": [1, 2, 3, 4, 5]}), keys="key"), np.array([1, 2, 3, 4, 5])
    )


def test_prepare_array_dataframe_2_cols() -> None:
    assert objects_are_equal(
        prepare_array(
            pl.DataFrame({"key1": [1, 2, 3, 4, 5], "key2": [1, 0, 1, 0, 1]}),
            keys=["key1", "key2"],
        ),
        np.array([[1, 1], [2, 0], [3, 1], [4, 0], [5, 1]]),
    )
