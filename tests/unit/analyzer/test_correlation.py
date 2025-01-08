from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import CorrelationAnalyzer
from arkas.output import CorrelationOutput, EmptyOutput, Output
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
    )


#########################################
#     Tests for CorrelationAnalyzer     #
#########################################


def test_correlation_analyzer_repr() -> None:
    assert repr(CorrelationAnalyzer(x="col1", y="col2")).startswith("CorrelationAnalyzer(")


def test_correlation_analyzer_str() -> None:
    assert str(CorrelationAnalyzer(x="col1", y="col2")).startswith("CorrelationAnalyzer(")


def test_correlation_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        CorrelationAnalyzer(x="col1", y="col2")
        .analyze(dataframe)
        .equal(
            CorrelationOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                )
            )
        )
    )


def test_correlation_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationAnalyzer(x="col1", y="col2").analyze(dataframe, lazy=False),
        Output,
    )


def test_correlation_analyzer_analyze_drop_nulls() -> None:
    assert (
        CorrelationAnalyzer(x="col1", y="col2")
        .analyze(
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 2, 1, 2, None, None],
                    "col2": [3, 2, 0, 1, 0, None, 1, None],
                    "col3": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            CorrelationOutput(
                DataFrameState(pl.DataFrame({"col1": [1, 2, 3, 2, 1], "col2": [3, 2, 0, 1, 0]}))
            )
        )
    )


def test_correlation_analyzer_analyze_drop_nulls_false() -> None:
    assert (
        CorrelationAnalyzer(x="col1", y="col2", drop_nulls=False)
        .analyze(
            pl.DataFrame(
                {
                    "col1": [1, 2, 3, 2, 1, 2, None, None],
                    "col2": [3, 2, 0, 1, 0, None, 1, None],
                }
            )
        )
        .equal(
            CorrelationOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1, 2, 3, 2, 1, 2, None, None],
                            "col2": [3, 2, 0, 1, 0, None, 1, None],
                        }
                    )
                )
            ),
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_correlation_analyzer_analyze_nan_policy(nan_policy: str) -> None:
    assert (
        CorrelationAnalyzer(x="col1", y="col2", nan_policy=nan_policy)
        .analyze(
            pl.DataFrame(
                {
                    "col2": [3, 2, 0, 1, 0, None],
                    "col1": [1, 2, 3, 2, 1, None],
                }
            )
        )
        .equal(
            CorrelationOutput(
                DataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                    nan_policy=nan_policy,
                ),
            )
        )
    )


def test_correlation_analyzer_analyze_missing_policy_ignore(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="m1", y="m2", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_correlation_analyzer_analyze_missing_ignore_x(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="missing", y="col2", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_correlation_analyzer_analyze_missing_ignore_y(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="col1", y="missing", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_correlation_analyzer_analyze_missing_policy_raise(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="m1", y="m2")
    with pytest.raises(ColumnNotFoundError, match="column 'm1' is missing in the DataFrame"):
        analyzer.analyze(dataframe)


def test_correlation_analyzer_analyze_missing_raise_x(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="missing", y="col2")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        analyzer.analyze(dataframe)


def test_correlation_analyzer_analyze_missing_raise_y(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="col1", y="missing")
    with pytest.raises(ColumnNotFoundError, match="column 'missing' is missing in the DataFrame"):
        analyzer.analyze(dataframe)


def test_correlation_analyzer_analyze_missing_policy_warn(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="m1", y="m2", missing_policy="warn")
    with (
        pytest.warns(
            ColumnNotFoundWarning,
            match="column 'm1' is missing in the DataFrame and will be ignored",
        ),
        pytest.warns(
            ColumnNotFoundWarning,
            match="column 'm2' is missing in the DataFrame and will be ignored",
        ),
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_correlation_analyzer_analyze_missing_warn_x(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="missing", y="col2", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignored",
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_correlation_analyzer_analyze_missing_warn_y(dataframe: pl.DataFrame) -> None:
    analyzer = CorrelationAnalyzer(x="col1", y="missing", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'missing' is missing in the DataFrame and will be ignored",
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())
