from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_allclose

from arkas.content import ContentGenerator, SummaryContentGenerator
from arkas.content.summary import create_table, create_table_row, create_template
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "float": [1.2, 4.2, None, 2.2, 1, 2.2],
            "int": [1, 1, 0, 1, 1, 1],
            "str": ["A", "B", None, None, "C", "B"],
        },
        schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    )


#############################################
#     Tests for SummaryContentGenerator     #
#############################################


def test_summary_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(SummaryContentGenerator(DataFrameState(dataframe))).startswith(
        "SummaryContentGenerator("
    )


def test_summary_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(SummaryContentGenerator(DataFrameState(dataframe))).startswith(
        "SummaryContentGenerator("
    )


def test_summary_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        SummaryContentGenerator(DataFrameState(dataframe)).compute(), ContentGenerator
    )


def test_summary_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert SummaryContentGenerator(DataFrameState(dataframe)).equal(
        SummaryContentGenerator(DataFrameState(dataframe))
    )


def test_summary_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not SummaryContentGenerator(DataFrameState(dataframe)).equal(
        SummaryContentGenerator(
            DataFrameState(
                pl.DataFrame(
                    {
                        "float": [1.2, 4.2, None, 2.2, 1, 2.2],
                        "int": [1, 1, 0, 1, 1, 1],
                    },
                    schema={"float": pl.Float64, "int": pl.Int64},
                )
            )
        )
    )


def test_summary_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not SummaryContentGenerator(DataFrameState(dataframe)).equal(42)


def test_summary_content_generator_get_columns(dataframe: pl.DataFrame) -> None:
    assert SummaryContentGenerator(DataFrameState(dataframe)).get_columns() == (
        "float",
        "int",
        "str",
    )


def test_summary_content_generator_get_columns_empty() -> None:
    assert SummaryContentGenerator(DataFrameState(pl.DataFrame({}))).get_columns() == ()


def test_summary_content_generator_get_null_count(dataframe: pl.DataFrame) -> None:
    assert SummaryContentGenerator(DataFrameState(dataframe)).get_null_count() == (1, 0, 2)


def test_summary_content_generator_get_null_count_empty() -> None:
    assert SummaryContentGenerator(DataFrameState(pl.DataFrame({}))).get_null_count() == ()


def test_summary_content_generator_get_nunique(dataframe: pl.DataFrame) -> None:
    assert SummaryContentGenerator(DataFrameState(dataframe)).get_nunique() == (5, 2, 4)


def test_summary_content_generator_get_nunique_empty() -> None:
    assert SummaryContentGenerator(DataFrameState(pl.DataFrame({}))).get_nunique() == ()


def test_summary_content_generator_get_dtypes(dataframe: pl.DataFrame) -> None:
    assert SummaryContentGenerator(DataFrameState(dataframe)).get_dtypes() == (
        pl.Float64(),
        pl.Int64(),
        pl.String(),
    )


def test_summary_content_generator_get_dtypes_empty() -> None:
    assert SummaryContentGenerator(DataFrameState(pl.DataFrame({}))).get_dtypes() == ()


def test_summary_content_generator_get_most_frequent_values(
    dataframe: pl.DataFrame,
) -> None:
    assert objects_are_allclose(
        SummaryContentGenerator(DataFrameState(dataframe)).get_most_frequent_values(),
        (
            ((2.2, 2), (1.2, 1), (4.2, 1), (None, 1), (1.0, 1)),
            ((1, 5), (0, 1)),
            (("B", 2), (None, 2), ("A", 1), ("C", 1)),
        ),
    )


def test_summary_content_generator_get_most_frequent_values_empty() -> None:
    assert (
        SummaryContentGenerator(DataFrameState(pl.DataFrame({}))).get_most_frequent_values() == ()
    )


def test_summary_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(SummaryContentGenerator(DataFrameState(dataframe)).generate_content(), str)


def test_summary_content_generator_generate_content_empty() -> None:
    assert isinstance(
        SummaryContentGenerator(DataFrameState(pl.DataFrame())).generate_content(), str
    )


def test_summary_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        SummaryContentGenerator(
            DataFrameState(
                pl.DataFrame(
                    {"float": [], "int": [], "str": []},
                    schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
                )
            )
        ).generate_content(),
        str,
    )


def test_summary_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(SummaryContentGenerator(DataFrameState(dataframe)).generate_body(), str)


def test_summary_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        SummaryContentGenerator(DataFrameState(dataframe)).generate_body(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


def test_summary_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(SummaryContentGenerator(DataFrameState(dataframe)).generate_toc(), str)


def test_summary_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        SummaryContentGenerator(DataFrameState(dataframe)).generate_toc(
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
            columns=["float", "int", "str"],
            null_count=(1, 0, 2),
            nunique=(5, 2, 4),
            dtypes=(pl.Float64(), pl.Int64(), pl.String()),
            most_frequent_values=(
                ((2.2, 2), (1.2, 1), (4.2, 1), (None, 1), (1.0, 1)),
                ((1, 5), (0, 1)),
                (("B", 2), (None, 2), ("A", 1), ("C", 1)),
            ),
            total=42,
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(
        create_table(
            columns=[],
            null_count=[],
            nunique=[],
            dtypes=[],
            most_frequent_values=[],
            total=0,
        ),
        str,
    )


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            null=5,
            nunique=42,
            dtype=pl.Float64(),
            most_frequent_values=[("C", 12), ("A", 5), ("B", 4)],
            total=100,
        ),
        str,
    )


def test_create_table_row_empty() -> None:
    assert isinstance(
        create_table_row(
            column="col",
            null=0,
            nunique=0,
            dtype=pl.Float64(),
            most_frequent_values=[],
            total=0,
        ),
        str,
    )
