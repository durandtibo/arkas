from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ColumnCooccurrenceContentGenerator, ContentGenerator
from arkas.content.column_cooccurrence import create_template
from arkas.figure import MatplotlibFigureConfig


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


########################################################
#     Tests for ColumnCooccurrenceContentGenerator     #
########################################################


def test_column_cooccurrence_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(ColumnCooccurrenceContentGenerator(dataframe)).startswith(
        "ColumnCooccurrenceContentGenerator("
    )


def test_column_cooccurrence_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(ColumnCooccurrenceContentGenerator(dataframe)).startswith(
        "ColumnCooccurrenceContentGenerator("
    )


def test_column_cooccurrence_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrenceContentGenerator(dataframe).compute(), ContentGenerator)


def test_column_cooccurrence_content_generator_frame(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceContentGenerator(dataframe).frame is dataframe


@pytest.mark.parametrize("ignore_self", [True, False])
def test_column_cooccurrence_content_generator_ignore_self(
    dataframe: pl.DataFrame, ignore_self: bool
) -> None:
    assert (
        ColumnCooccurrenceContentGenerator(dataframe, ignore_self=ignore_self).ignore_self
        == ignore_self
    )


def test_column_cooccurrence_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceContentGenerator(dataframe).equal(
        ColumnCooccurrenceContentGenerator(dataframe)
    )


def test_column_cooccurrence_content_generator_equal_false_different_frame(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceContentGenerator(dataframe).equal(
        ColumnCooccurrenceContentGenerator(
            pl.DataFrame(
                {
                    "float": [1.2, 4.2, None, 2.2, 1, 2.2],
                    "int": [1, 1, 0, 1, 1, 1],
                },
                schema={"float": pl.Float64, "int": pl.Int64},
            )
        )
    )


def test_column_cooccurrence_content_generator_equal_false_different_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceContentGenerator(dataframe).equal(
        ColumnCooccurrenceContentGenerator(dataframe, ignore_self=True)
    )


def test_column_cooccurrence_content_generator_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceContentGenerator(dataframe).equal(
        ColumnCooccurrenceContentGenerator(dataframe, figure_config=MatplotlibFigureConfig(dpi=50))
    )


def test_column_cooccurrence_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceContentGenerator(dataframe).equal(42)


def test_column_cooccurrence_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrenceContentGenerator(dataframe).generate_content(), str)


def test_column_cooccurrence_content_generator_generate_content_empty() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(frame=pl.DataFrame()).generate_content(), str
    )


def test_column_cooccurrence_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            frame=pl.DataFrame(
                {"float": [], "int": [], "str": []},
                schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
            )
        ).generate_content(),
        str,
    )


def test_column_cooccurrence_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrenceContentGenerator(dataframe).generate_body(), str)


def test_column_cooccurrence_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(dataframe).generate_body(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


def test_column_cooccurrence_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrenceContentGenerator(dataframe).generate_toc(), str)


def test_column_cooccurrence_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(dataframe).generate_toc(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
