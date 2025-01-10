from __future__ import annotations

import warnings

import polars as pl
import pytest
from coola import objects_are_equal
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import ColumnCorrelationAnalyzer
from arkas.output import ColumnCorrelationOutput, EmptyOutput, Output
from arkas.state import TargetDataFrameState


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


###############################################
#     Tests for ColumnCorrelationAnalyzer     #
###############################################


def test_numeric_summary_analyzer_repr() -> None:
    assert repr(ColumnCorrelationAnalyzer(target_column="col3")).startswith(
        "ColumnCorrelationAnalyzer("
    )


def test_numeric_summary_analyzer_str() -> None:
    assert str(ColumnCorrelationAnalyzer(target_column="col3")).startswith(
        "ColumnCorrelationAnalyzer("
    )


def test_numeric_summary_analyzer_analyze(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col3")
        .analyze(dataframe)
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(dataframe, target_column="col3", sort_key="spearman_coeff")
            )
        )
    )


def test_numeric_summary_analyzer_analyze_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationAnalyzer(target_column="col3").analyze(dataframe, lazy=False), Output
    )


def test_numeric_summary_analyzer_analyze_ignore_non_numeric_columns(
    dataframe: pl.DataFrame,
) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col3")
        .analyze(dataframe.with_columns(pl.lit("abc").alias("col4")))
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(dataframe, target_column="col3", sort_key="spearman_coeff")
            )
        )
    )


def test_numeric_summary_analyzer_analyze_sort_key(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col3", sort_key="pearson_coeff")
        .analyze(dataframe)
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(dataframe, target_column="col3", sort_key="pearson_coeff")
            )
        )
    )


def test_numeric_summary_analyzer_analyze_columns(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col2", columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                    target_column="col2",
                    sort_key="spearman_coeff",
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_columns_add_target_column(
    dataframe: pl.DataFrame,
) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col3", columns=["col1", "col2"])
        .analyze(dataframe)
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(dataframe, target_column="col3", sort_key="spearman_coeff")
            )
        )
    )


def test_numeric_summary_analyzer_analyze_exclude_columns(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCorrelationAnalyzer(target_column="col2", exclude_columns=["col3"])
        .analyze(dataframe)
        .equal(
            ColumnCorrelationOutput(
                TargetDataFrameState(
                    pl.DataFrame(
                        {
                            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                        },
                        schema={"col1": pl.Float64, "col2": pl.Float64},
                    ),
                    target_column="col2",
                    sort_key="spearman_coeff",
                )
            )
        )
    )


def test_numeric_summary_analyzer_analyze_missing_policy_ignore(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(
        target_column="col3", columns=["col1", "col2", "col3", "col5"], missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(
        ColumnCorrelationOutput(
            TargetDataFrameState(dataframe, target_column="col3", sort_key="spearman_coeff")
        )
    )


def test_numeric_summary_analyzer_analyze_missing_policy_ignore_target_column(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(
        target_column="col5", columns=["col1", "col2", "col3"], missing_policy="ignore"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_numeric_summary_analyzer_analyze_missing_policy_raise(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(
        target_column="col3", columns=["col1", "col2", "col3", "col5"]
    )
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_numeric_summary_analyzer_analyze_missing_policy_raise_target_column(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(target_column="col5", columns=["col1", "col2", "col3"])
    with pytest.raises(ColumnNotFoundError, match="1 column is missing in the DataFrame:"):
        analyzer.analyze(dataframe)


def test_numeric_summary_analyzer_analyze_missing_policy_warn(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(
        target_column="col3", columns=["col1", "col2", "col3", "col5"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(
        ColumnCorrelationOutput(
            TargetDataFrameState(dataframe, target_column="col3", sort_key="spearman_coeff")
        )
    )


def test_numeric_summary_analyzer_analyze_missing_policy_warn_target_column(
    dataframe: pl.DataFrame,
) -> None:
    analyzer = ColumnCorrelationAnalyzer(
        target_column="col5", columns=["col1", "col2", "col3"], missing_policy="warn"
    )
    with pytest.warns(
        ColumnNotFoundWarning, match="1 column is missing in the DataFrame and will be ignored:"
    ):
        out = analyzer.analyze(dataframe)
    assert out.equal(EmptyOutput())


def test_numeric_summary_analyzer_equal_true() -> None:
    assert ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col3")
    )


def test_numeric_summary_analyzer_equal_false_different_target_column() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col1")
    )


def test_numeric_summary_analyzer_equal_false_different_columns() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col3", columns=["col1", "col2"])
    )


def test_numeric_summary_analyzer_equal_false_different_exclude_columns() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col3", exclude_columns=["col2"])
    )


def test_numeric_summary_analyzer_equal_false_different_missing_policy() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col3", missing_policy="warn")
    )


def test_numeric_summary_analyzer_equal_false_different_sort_key() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(
        ColumnCorrelationAnalyzer(target_column="col3", sort_key="pearson_coeff")
    )


def test_numeric_summary_analyzer_equal_false_different_type() -> None:
    assert not ColumnCorrelationAnalyzer(target_column="col3").equal(42)


def test_numeric_summary_analyzer_get_args() -> None:
    assert objects_are_equal(
        ColumnCorrelationAnalyzer(target_column="col3").get_args(),
        {
            "target_column": "col3",
            "columns": None,
            "exclude_columns": (),
            "missing_policy": "raise",
            "sort_key": "spearman_coeff",
        },
    )
