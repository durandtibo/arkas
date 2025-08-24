from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.analyzer import BaseInNLazyAnalyzer
from arkas.output import EmptyOutput


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["1", "2", "3", "4", "5"],
            "col3": ["a ", " b", "  c  ", "d", "e"],
            "col4": ["a ", " b", "  c  ", "d", "e"],
        }
    )


#########################################
#     Tests for BaseInNLazyAnalyzer     #
#########################################


class MyInNLazyAnalyzer(BaseInNLazyAnalyzer):
    def _analyze(self, frame: pl.DataFrame) -> EmptyOutput:  # noqa: ARG002
        return EmptyOutput()


def test_base_in_n_lazy_analyzer_repr() -> None:
    assert (
        repr(MyInNLazyAnalyzer())
        == "MyInNLazyAnalyzer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_base_in_n_lazy_analyzer_str() -> None:
    assert (
        str(MyInNLazyAnalyzer())
        == "MyInNLazyAnalyzer(columns=None, exclude_columns=(), missing_policy='raise')"
    )


def test_base_in_n_lazy_analyzer_find_columns(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer(columns=["col2", "col3", "col5"]).find_columns(dataframe) == (
        "col2",
        "col3",
        "col5",
    )


def test_base_in_n_lazy_analyzer_find_columns_none(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer().find_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_base_in_n_lazy_analyzer_find_columns_exclude(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer(
        columns=["col2", "col3", "col5"], exclude_columns=["col3", "col6"]
    ).find_columns(dataframe) == ("col2", "col5")


def test_base_in_n_lazy_analyzer_find_common_columns(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer(columns=["col2", "col3", "col5"]).find_common_columns(dataframe) == (
        "col2",
        "col3",
    )


def test_base_in_n_lazy_analyzer_find_common_columns_none(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer().find_common_columns(dataframe) == ("col1", "col2", "col3", "col4")


def test_base_in_n_lazy_analyzer_find_common_columns_exclude(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer(exclude_columns=["col3", "col6"]).find_common_columns(dataframe) == (
        "col1",
        "col2",
        "col4",
    )


def test_base_in_n_lazy_analyzer_find_missing_columns(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer(columns=["col2", "col3", "col5"]).find_missing_columns(dataframe) == (
        "col5",
    )


def test_base_in_n_lazy_analyzer_find_missing_columns_none(dataframe: pl.DataFrame) -> None:
    assert MyInNLazyAnalyzer().find_missing_columns(dataframe) == ()


def test_base_in_n_lazy_analyzer_get_args() -> None:
    assert objects_are_equal(
        MyInNLazyAnalyzer().get_args(),
        {"columns": None, "exclude_columns": (), "missing_policy": "raise"},
    )
