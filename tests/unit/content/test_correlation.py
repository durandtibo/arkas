from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, CorrelationContentGenerator
from arkas.content.correlation import create_template
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64},
    )


#################################################
#     Tests for CorrelationContentGenerator     #
#################################################


def test_correlation_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(CorrelationContentGenerator(DataFrameState(dataframe))).startswith(
        "CorrelationContentGenerator("
    )


def test_correlation_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(CorrelationContentGenerator(DataFrameState(dataframe))).startswith(
        "CorrelationContentGenerator("
    )


def test_correlation_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).compute(),
        ContentGenerator,
    )


def test_correlation_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert CorrelationContentGenerator(DataFrameState(dataframe)).equal(
        CorrelationContentGenerator(DataFrameState(dataframe))
    )


def test_correlation_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not CorrelationContentGenerator(DataFrameState(dataframe)).equal(
        CorrelationContentGenerator(
            DataFrameState(
                pl.DataFrame(
                    {"col1": [], "col2": []},
                    schema={"col1": pl.Float64, "col2": pl.Float64},
                ),
            )
        )
    )


def test_correlation_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not CorrelationContentGenerator(DataFrameState(dataframe)).equal(42)


def test_correlation_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).generate_content(),
        str,
    )


def test_correlation_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        CorrelationContentGenerator(
            DataFrameState(
                pl.DataFrame(
                    {"col1": [], "col2": []},
                    schema={"col1": pl.Float64, "col2": pl.Float64},
                ),
            )
        ).generate_content(),
        str,
    )


def test_correlation_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).generate_body(),
        str,
    )


def test_correlation_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).generate_body(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


def test_correlation_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).generate_toc(),
        str,
    )


def test_correlation_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        CorrelationContentGenerator(DataFrameState(dataframe)).generate_toc(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
