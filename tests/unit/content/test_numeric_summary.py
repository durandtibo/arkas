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
                    schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
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


def test_create_table(dataframe: pl.DataFrame) -> None:
    assert isinstance(create_table(dataframe), str)


def test_create_table_empty() -> None:
    assert isinstance(create_table(pl.DataFrame()), str)


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(create_table_row(pl.Series("col", [1, 2, 3, 4, 5, 6, 7])), str)


def test_create_table_row_empty() -> None:
    assert isinstance(create_table_row(pl.Series("col", [])), str)


############################################
#     Tests for create_table_quantiles     #
############################################


def test_create_table_quantiles(dataframe: pl.DataFrame) -> None:
    assert isinstance(create_table_quantiles(dataframe), str)


def test_create_table_quantiles_empty() -> None:
    assert isinstance(create_table_quantiles(pl.DataFrame()), str)


################################################
#     Tests for create_table_quantiles_row     #
################################################


def test_create_table_quantiles_row() -> None:
    assert isinstance(create_table_quantiles_row(pl.Series("col", [1, 2, 3, 4, 5, 6, 7])), str)


def test_create_table_quantiles_row_empty() -> None:
    assert isinstance(create_table_quantiles_row(pl.Series("col", [])), str)
