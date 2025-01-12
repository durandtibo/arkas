from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, NumericSummaryContentGenerator
from arkas.content.numeric_summary import (
    create_table,
    create_table_quantiles,
    create_table_quantiles_row,
    create_table_row,
    create_template,
)
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    )


####################################################
#     Tests for NumericSummaryContentGenerator     #
####################################################


def test_numeric_summary_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(NumericSummaryContentGenerator(DataFrameState(dataframe))).startswith(
        "NumericSummaryContentGenerator("
    )


def test_numeric_summary_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(NumericSummaryContentGenerator(DataFrameState(dataframe))).startswith(
        "NumericSummaryContentGenerator("
    )


def test_numeric_summary_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(dataframe)).compute(), ContentGenerator
    )


def test_numeric_summary_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert NumericSummaryContentGenerator(DataFrameState(dataframe)).equal(
        NumericSummaryContentGenerator(DataFrameState(dataframe))
    )


def test_numeric_summary_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not NumericSummaryContentGenerator(DataFrameState(dataframe)).equal(
        NumericSummaryContentGenerator(DataFrameState(pl.DataFrame()))
    )


def test_numeric_summary_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not NumericSummaryContentGenerator(DataFrameState(dataframe)).equal(42)


def test_numeric_summary_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(dataframe)).generate_content(), str
    )


def test_numeric_summary_content_generator_generate_content_empty() -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(pl.DataFrame())).generate_content(), str
    )


def test_numeric_summary_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        NumericSummaryContentGenerator(
            DataFrameState(
                pl.DataFrame(
                    {"float": [], "int": [], "str": []},
                    schema={"float": pl.Float64, "int": pl.Int64, "str": pl.Float64},
                )
            )
        ).generate_content(),
        str,
    )


def test_numeric_summary_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(dataframe)).generate_body(), str
    )


def test_numeric_summary_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(dataframe)).generate_body(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


def test_numeric_summary_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(NumericSummaryContentGenerator(DataFrameState(dataframe)).generate_toc(), str)


def test_numeric_summary_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        NumericSummaryContentGenerator(DataFrameState(dataframe)).generate_toc(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(
        create_table(
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
            }
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(create_table({}), str)


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            metrics={
                "count": 101,
                "num_nulls": 0,
                "num_nans": 0,
                "nunique": 101,
                "mean": 50.0,
                "std": 29.0,
                "skewness": 0.0,
                "kurtosis": -1.2,
                "min": 0.0,
                "q001": 0.1,
                "q01": 1.0,
                "q05": 5.0,
                "q10": 10.0,
                "q25": 25.0,
                "median": 50.0,
                "q75": 75.0,
                "q90": 90.0,
                "q95": 95.0,
                "q99": 99.0,
                "q999": 99.9,
                "max": 100.0,
                ">0": 100,
                "<0": 0,
                "=0": 1,
            },
        ),
        str,
    )


def test_create_table_row_empty() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            metrics={
                "count": 0,
                "num_nulls": 0,
                "num_nans": 0,
                "nunique": 0,
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
        ),
        str,
    )


############################################
#     Tests for create_table_quantiles     #
############################################


def test_create_table_quantiles() -> None:
    assert isinstance(
        create_table_quantiles(
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
            }
        ),
        str,
    )


def test_create_table_quantiles_empty() -> None:
    assert isinstance(create_table_quantiles({}), str)


################################################
#     Tests for create_table_quantiles_row     #
################################################


def test_create_table_quantiles_row() -> None:
    assert isinstance(
        create_table_quantiles_row(
            column="col",
            metrics={
                "count": 101,
                "num_nulls": 0,
                "num_nans": 0,
                "nunique": 101,
                "mean": 50.0,
                "std": 29.0,
                "skewness": 0.0,
                "kurtosis": -1.2,
                "min": 0.0,
                "q001": 0.1,
                "q01": 1.0,
                "q05": 5.0,
                "q10": 10.0,
                "q25": 25.0,
                "median": 50.0,
                "q75": 75.0,
                "q90": 90.0,
                "q95": 95.0,
                "q99": 99.0,
                "q999": 99.9,
                "max": 100.0,
                ">0": 100,
                "<0": 0,
                "=0": 1,
            },
        ),
        str,
    )


def test_create_table_quantiles_row_empty() -> None:
    assert isinstance(
        create_table_quantiles_row(
            column="col",
            metrics={
                "count": 0,
                "num_nulls": 0,
                "num_nans": 0,
                "nunique": 0,
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
        ),
        str,
    )
