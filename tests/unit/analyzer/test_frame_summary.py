from __future__ import annotations

import polars as pl
import pytest

from arkas.analyzer import DataFrameSummaryAnalyzer
from arkas.output import DataFrameSummaryOutput

##############################################
#     Tests for DataFrameSummaryAnalyzer     #
##############################################


def test_dataframe_summary_analyzer_repr() -> None:
    assert repr(DataFrameSummaryAnalyzer()).startswith("DataFrameSummaryAnalyzer(")


def test_dataframe_summary_analyzer_str() -> None:
    assert str(DataFrameSummaryAnalyzer()).startswith("DataFrameSummaryAnalyzer(")


def test_dataframe_summary_analyzer_analyze() -> None:
    assert (
        DataFrameSummaryAnalyzer()
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(
            DataFrameSummaryOutput(
                pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]})
            )
        )
    )


@pytest.mark.parametrize("top", [0, 1, 2])
def test_dataframe_summary_analyzer_analyze_top(top: int) -> None:
    assert (
        DataFrameSummaryAnalyzer(top=top)
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(
            DataFrameSummaryOutput(
                pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}), top=top
            )
        )
    )


def test_dataframe_summary_analyzer_analyze_incorrect_top() -> None:
    with pytest.raises(ValueError, match=r"Incorrect top value \(-1\). top must be positive"):
        DataFrameSummaryAnalyzer(top=-1)


def test_dataframe_summary_analyzer_analyze_sort_true() -> None:
    assert (
        DataFrameSummaryAnalyzer(sort=True)
        .analyze(pl.DataFrame({"col2": [3, 2, 0, 1, 0], "col1": [1, 2, 3, 2, 1]}))
        .equal(
            DataFrameSummaryOutput(pl.DataFrame({"col1": [1, 2, 3, 2, 1], "col2": [3, 2, 0, 1, 0]}))
        )
    )
