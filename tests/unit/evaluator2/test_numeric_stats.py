from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose

from arkas.evaluator2 import Evaluator, NumericStatisticsEvaluator
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


################################################
#     Tests for NumericStatisticsEvaluator     #
################################################


def test_numeric_statistics_evaluator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(NumericStatisticsEvaluator(DataFrameState(dataframe))).startswith(
        "NumericStatisticsEvaluator("
    )


def test_numeric_statistics_evaluator_str(dataframe: pl.DataFrame) -> None:
    assert str(NumericStatisticsEvaluator(DataFrameState(dataframe))).startswith(
        "NumericStatisticsEvaluator("
    )


def test_numeric_statistics_evaluator_state(dataframe: pl.DataFrame) -> None:
    assert NumericStatisticsEvaluator(DataFrameState(dataframe)).state.equal(
        DataFrameState(dataframe)
    )


def test_numeric_statistics_evaluator_equal_true(dataframe: pl.DataFrame) -> None:
    assert NumericStatisticsEvaluator(DataFrameState(dataframe)).equal(
        NumericStatisticsEvaluator(DataFrameState(dataframe))
    )


def test_numeric_statistics_evaluator_equal_false_different_state(dataframe: pl.DataFrame) -> None:
    assert not NumericStatisticsEvaluator(DataFrameState(dataframe)).equal(
        NumericStatisticsEvaluator(DataFrameState(pl.DataFrame({"col1": [], "col2": []})))
    )


def test_numeric_statistics_evaluator_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not NumericStatisticsEvaluator(DataFrameState(dataframe)).equal(42)


def test_numeric_statistics_evaluator_evaluate(dataframe: pl.DataFrame) -> None:
    evaluator = NumericStatisticsEvaluator(DataFrameState(dataframe))
    assert objects_are_allclose(
        evaluator.evaluate(),
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


def test_numeric_statistics_evaluator_evaluate_one_row() -> None:
    evaluator = NumericStatisticsEvaluator(
        DataFrameState(
            pl.DataFrame(
                {"col1": [1.0], "col2": [7.0]}, schema={"col1": pl.Float64, "col2": pl.Float64}
            )
        )
    )
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "col1": {
                "count": 1,
                "nunique": 1,
                "num_nans": 0,
                "num_nulls": 0,
                "mean": 1.0,
                "std": 0.0,
                "skewness": float("nan"),
                "kurtosis": float("nan"),
                "min": 1.0,
                "q001": 1.0,
                "q01": 1.0,
                "q05": 1.0,
                "q10": 1.0,
                "q25": 1.0,
                "median": 1.0,
                "q75": 1.0,
                "q90": 1.0,
                "q95": 1.0,
                "q99": 1.0,
                "q999": 1.0,
                "max": 1.0,
                ">0": 1,
                "<0": 0,
                "=0": 0,
            },
            "col2": {
                "count": 1,
                "nunique": 1,
                "num_nans": 0,
                "num_nulls": 0,
                "mean": 7.0,
                "std": 0.0,
                "skewness": float("nan"),
                "kurtosis": float("nan"),
                "min": 7.0,
                "q001": 7.0,
                "q01": 7.0,
                "q05": 7.0,
                "q10": 7.0,
                "q25": 7.0,
                "median": 7.0,
                "q75": 7.0,
                "q90": 7.0,
                "q95": 7.0,
                "q99": 7.0,
                "q999": 7.0,
                "max": 7.0,
                ">0": 1,
                "<0": 0,
                "=0": 0,
            },
        },
        equal_nan=True,
    )


def test_numeric_statistics_evaluator_evaluate_empty() -> None:
    evaluator = NumericStatisticsEvaluator(DataFrameState(pl.DataFrame({"col1": [], "col2": []})))
    assert objects_are_allclose(
        evaluator.evaluate(),
        {
            "col1": {
                "count": 0,
                "nunique": 0,
                "num_nans": 0,
                "num_nulls": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "skewness": float("nan"),
                "kurtosis": float("nan"),
                "min": float("nan"),
                "q001": float("nan"),
                "q01": float("nan"),
                "q05": float("nan"),
                "q10": float("nan"),
                "q25": float("nan"),
                "median": float("nan"),
                "q75": float("nan"),
                "q90": float("nan"),
                "q95": float("nan"),
                "q99": float("nan"),
                "q999": float("nan"),
                "max": float("nan"),
                ">0": 0,
                "<0": 0,
                "=0": 0,
            },
            "col2": {
                "count": 0,
                "nunique": 0,
                "num_nans": 0,
                "num_nulls": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "skewness": float("nan"),
                "kurtosis": float("nan"),
                "min": float("nan"),
                "q001": float("nan"),
                "q01": float("nan"),
                "q05": float("nan"),
                "q10": float("nan"),
                "q25": float("nan"),
                "median": float("nan"),
                "q75": float("nan"),
                "q90": float("nan"),
                "q95": float("nan"),
                "q99": float("nan"),
                "q999": float("nan"),
                "max": float("nan"),
                ">0": 0,
                "<0": 0,
                "=0": 0,
            },
        },
        equal_nan=True,
    )


def test_numeric_statistics_evaluator_evaluate_prefix_suffix(dataframe: pl.DataFrame) -> None:
    evaluator = NumericStatisticsEvaluator(DataFrameState(dataframe))
    assert objects_are_allclose(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_col1_suffix": {
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
            "prefix_col2_suffix": {
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
            "prefix_col3_suffix": {
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


def test_numeric_statistics_evaluator_compute(dataframe: pl.DataFrame) -> None:
    assert (
        NumericStatisticsEvaluator(DataFrameState(dataframe))
        .compute()
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
