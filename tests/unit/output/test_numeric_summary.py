from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, NumericSummaryContentGenerator
from arkas.evaluator2 import Evaluator, NumericStatisticsEvaluator
from arkas.output import NumericSummaryOutput, Output
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": [1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
    )


##########################################
#     Tests for NumericSummaryOutput     #
##########################################


def test_numeric_summary_output_repr(dataframe: pl.DataFrame) -> None:
    assert repr(NumericSummaryOutput(DataFrameState(dataframe))).startswith("NumericSummaryOutput(")


def test_numeric_summary_output_str(dataframe: pl.DataFrame) -> None:
    assert str(NumericSummaryOutput(DataFrameState(dataframe))).startswith("NumericSummaryOutput(")


def test_numeric_summary_output_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(NumericSummaryOutput(DataFrameState(dataframe)).compute(), Output)


def test_numeric_summary_output_equal_true(dataframe: pl.DataFrame) -> None:
    assert NumericSummaryOutput(DataFrameState(dataframe)).equal(
        NumericSummaryOutput(DataFrameState(dataframe))
    )


def test_numeric_summary_output_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not NumericSummaryOutput(DataFrameState(dataframe)).equal(DataFrameState(pl.DataFrame()))


def test_numeric_summary_output_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not NumericSummaryOutput(DataFrameState(dataframe)).equal(42)


def test_numeric_summary_output_get_content_generator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryOutput(DataFrameState(dataframe))
        .get_content_generator()
        .equal(NumericSummaryContentGenerator(DataFrameState(dataframe)))
    )


def test_numeric_summary_output_get_content_generator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryOutput(DataFrameState(dataframe)).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_numeric_summary_output_get_evaluator_lazy_true(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryOutput(DataFrameState(dataframe))
        .get_evaluator()
        .equal(NumericStatisticsEvaluator(DataFrameState(dataframe)))
    )


def test_numeric_summary_output_get_evaluator_lazy_false(dataframe: pl.DataFrame) -> None:
    assert (
        NumericSummaryOutput(DataFrameState(dataframe))
        .get_evaluator(lazy=False)
        .allclose(
            Evaluator(
                {
                    "col1": {
                        "count": 7,
                        "nunique": 7,
                        "num_nans": 0,
                        "num_nulls": 0,
                        "mean": 4.0,
                        "std": 2.0,
                        "skewness": 0.0,
                        "kurtosis": -1.25,
                        "min": 1.0,
                        "q001": 1.006,
                        "q01": 1.06,
                        "q05": 1.3,
                        "q10": 1.6,
                        "q25": 2.5,
                        "median": 4.0,
                        "q75": 5.5,
                        "q90": 6.4,
                        "q95": 6.7,
                        "q99": 6.94,
                        "q999": 6.994,
                        "max": 7.0,
                        ">0": 7,
                        "<0": 0,
                        "=0": 0,
                    },
                    "col2": {
                        "count": 7,
                        "nunique": 7,
                        "num_nans": 0,
                        "num_nulls": 0,
                        "mean": 4.0,
                        "std": 2.0,
                        "skewness": 0.0,
                        "kurtosis": -1.25,
                        "min": 1.0,
                        "q001": 1.006,
                        "q01": 1.06,
                        "q05": 1.3,
                        "q10": 1.6,
                        "q25": 2.5,
                        "median": 4.0,
                        "q75": 5.5,
                        "q90": 6.4,
                        "q95": 6.7,
                        "q99": 6.94,
                        "q999": 6.994,
                        "max": 7.0,
                        ">0": 7,
                        "<0": 0,
                        "=0": 0,
                    },
                    "col3": {
                        "count": 7,
                        "nunique": 3,
                        "num_nans": 0,
                        "num_nulls": 0,
                        "mean": 2.0,
                        "std": 0.7559289460184544,
                        "skewness": 0.0,
                        "kurtosis": -1.25,
                        "min": 1.0,
                        "q001": 1.0,
                        "q01": 1.0,
                        "q05": 1.0,
                        "q10": 1.0,
                        "q25": 1.5,
                        "median": 2.0,
                        "q75": 2.5,
                        "q90": 3.0,
                        "q95": 3.0,
                        "q99": 3.0,
                        "q999": 3.0,
                        "max": 3.0,
                        ">0": 7,
                        "<0": 0,
                        "=0": 0,
                    },
                },
            )
        )
    )
