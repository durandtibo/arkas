from __future__ import annotations

import warnings

import polars as pl
import pytest
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import CorrelationAnalyzer
from arkas.figure import MatplotlibFigureConfig
from arkas.output import CorrelationOutput, EmptyOutput, Output
from arkas.state import TwoColumnDataFrameState


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
                TwoColumnDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                    column1="col1",
                    column2="col2",
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
            ),
        )
        .equal(
            CorrelationOutput(
                TwoColumnDataFrameState(
                    pl.DataFrame({"col1": [1, 2, 3, 2, 1], "col2": [3, 2, 0, 1, 0]}),
                    column1="col1",
                    column2="col2",
                )
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
                TwoColumnDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1, 2, 3, 2, 1, 2, None, None],
                            "col2": [3, 2, 0, 1, 0, None, 1, None],
                        }
                    ),
                    column1="col1",
                    column2="col2",
                )
            ),
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_correlation_analyzer_analyze_nan_policy(dataframe: pl.DataFrame, nan_policy: str) -> None:
    assert (
        CorrelationAnalyzer(x="col1", y="col2", nan_policy=nan_policy)
        .analyze(dataframe)
        .equal(
            CorrelationOutput(
                TwoColumnDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                    column1="col1",
                    column2="col2",
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


def test_correlation_analyzer_equal_true() -> None:
    assert CorrelationAnalyzer(x="col1", y="col2").equal(CorrelationAnalyzer(x="col1", y="col2"))


def test_correlation_analyzer_equal_false_different_x() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(CorrelationAnalyzer(x="col", y="col2"))


def test_correlation_analyzer_equal_false_different_y() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(CorrelationAnalyzer(x="col1", y="col"))


def test_correlation_analyzer_equal_false_different_drop_nulls() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(
        CorrelationAnalyzer(x="col1", y="col2", drop_nulls=False)
    )


def test_correlation_analyzer_equal_false_different_missing_policy() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(
        CorrelationAnalyzer(x="col1", y="col2", missing_policy="warn")
    )


def test_correlation_analyzer_equal_false_different_nan_policy() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(
        CorrelationAnalyzer(x="col1", y="col2", nan_policy="raise")
    )


def test_correlation_analyzer_equal_false_different_figure_config() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(
        CorrelationAnalyzer(x="col1", y="col2", figure_config=MatplotlibFigureConfig())
    )


def test_correlation_analyzer_equal_false_different_type() -> None:
    assert not CorrelationAnalyzer(x="col1", y="col2").equal(42)
