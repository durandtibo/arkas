from __future__ import annotations

import polars as pl
import pytest

from arkas.analyzer import SummaryAnalyzer
from arkas.output import Output, SummaryOutput
from arkas.state import DataFrameState

#####################################
#     Tests for SummaryAnalyzer     #
#####################################


def test_summary_analyzer_analyze_incorrect_top() -> None:
    with pytest.raises(ValueError, match=r"Incorrect 'top': -1. The value must be positive"):
        SummaryAnalyzer(top=-1)


def test_summary_analyzer_repr() -> None:
    assert repr(SummaryAnalyzer()).startswith("SummaryAnalyzer(")


def test_summary_analyzer_str() -> None:
    assert str(SummaryAnalyzer()).startswith("SummaryAnalyzer(")


def test_summary_analyzer_analyze() -> None:
    assert (
        SummaryAnalyzer()
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(
            SummaryOutput(
                DataFrameState(
                    pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}), top=5
                )
            )
        )
    )


def test_summary_analyzer_analyze_lazy_false() -> None:
    assert isinstance(
        SummaryAnalyzer().analyze(
            pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}), lazy=False
        ),
        Output,
    )


@pytest.mark.parametrize("top", [0, 1, 2])
def test_summary_analyzer_analyze_top(top: int) -> None:
    assert (
        SummaryAnalyzer(top=top)
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(
            SummaryOutput(
                DataFrameState(
                    pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}), top=top
                )
            )
        )
    )
