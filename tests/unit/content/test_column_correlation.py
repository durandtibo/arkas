from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ColumnCorrelationContentGenerator, ContentGenerator
from arkas.content.column_correlation import (
    create_table,
    create_table_row,
    create_template,
)
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


#######################################################
#     Tests for ColumnCorrelationContentGenerator     #
#######################################################


def test_column_correlation_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ColumnCorrelationContentGenerator(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationContentGenerator(")


def test_column_correlation_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnCorrelationContentGenerator(TargetDataFrameState(dataframe, target_column="col3"))
    ).startswith("ColumnCorrelationContentGenerator(")


def test_column_correlation_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).compute(),
        ContentGenerator,
    )


def test_column_correlation_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCorrelationContentGenerator(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(
        ColumnCorrelationContentGenerator(TargetDataFrameState(dataframe, target_column="col3"))
    )


def test_column_correlation_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCorrelationContentGenerator(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(
        ColumnCorrelationContentGenerator(TargetDataFrameState(dataframe, target_column="col1"))
    )


def test_column_correlation_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCorrelationContentGenerator(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(42)


def test_column_correlation_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).generate_content(),
        str,
    )


def test_column_correlation_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(
                pl.DataFrame(
                    {"col1": [], "col2": [], "col3": []},
                    schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
                ),
                target_column="col3",
            )
        ).generate_content(),
        str,
    )


def test_column_correlation_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).generate_body(),
        str,
    )


def test_column_correlation_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_column_correlation_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).generate_toc(),
        str,
    )


def test_column_correlation_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            TargetDataFrameState(dataframe, target_column="col3")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
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
    assert isinstance(create_table_row(pl.Series("col", [], dtype=pl.Float64)), str)
