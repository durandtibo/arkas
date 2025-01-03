from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.content import ColumnCooccurrenceContentGenerator, ContentGenerator
from arkas.content.column_cooccurrence import (
    create_table,
    create_table_row,
    create_table_section,
    create_template,
)
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


##########################################
#     Tests for create_table_section     #
##########################################


def test_create_table_section() -> None:
    assert isinstance(
        create_table_section(
            matrix=np.array([[5, 7, 1], [0, 6, 3], [8, 2, 4]]), columns=["col1", "col2", "col3"]
        ),
        str,
    )


def test_create_table_section_empty() -> None:
    assert isinstance(create_table_section(matrix=np.zeros((0, 0)), columns=[]), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert create_table(
        np.array([[80, 4, 6], [4, 2, 5], [6, 5, 3]]), columns=["col1", "col2", "col3"]
    ) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        '        <tr><th>1</th><td>col1</td><td>col1</td><td style="text-align: right;">80</td><td style="text-align: right;">80.0000 %</td></tr>\n'
        '        <tr><th>2</th><td>col3</td><td>col1</td><td style="text-align: right;">6</td><td style="text-align: right;">6.0000 %</td></tr>\n'
        '        <tr><th>3</th><td>col3</td><td>col2</td><td style="text-align: right;">5</td><td style="text-align: right;">5.0000 %</td></tr>\n'
        '        <tr><th>4</th><td>col2</td><td>col1</td><td style="text-align: right;">4</td><td style="text-align: right;">4.0000 %</td></tr>\n'
        '        <tr><th>5</th><td>col3</td><td>col3</td><td style="text-align: right;">3</td><td style="text-align: right;">3.0000 %</td></tr>\n'
        '        <tr><th>6</th><td>col2</td><td>col2</td><td style="text-align: right;">2</td><td style="text-align: right;">2.0000 %</td></tr>\n'
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


def test_create_table_top_2() -> None:
    assert create_table(
        np.array([[80, 4, 6], [4, 2, 5], [6, 5, 3]]), columns=["col1", "col2", "col3"], top=2
    ) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        '        <tr><th>1</th><td>col1</td><td>col1</td><td style="text-align: right;">80</td><td style="text-align: right;">80.0000 %</td></tr>\n'
        '        <tr><th>2</th><td>col3</td><td>col1</td><td style="text-align: right;">6</td><td style="text-align: right;">6.0000 %</td></tr>\n'
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


def test_create_table_empty() -> None:
    assert create_table(matrix=np.zeros((0, 0)), columns=[]) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        "        \n"
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert create_table_row(rank=2, col1="cat", col2="meow", count=42, total=100) == (
        '<tr><th>2</th><td>cat</td><td>meow</td><td style="text-align: right;">42</td>'
        '<td style="text-align: right;">42.0000 %</td></tr>'
    )


def test_create_table_row_empty() -> None:
    assert create_table_row(rank=2, col1="cat", col2="meow", count=0, total=0) == (
        '<tr><th>2</th><td>cat</td><td>meow</td><td style="text-align: right;">0</td>'
        '<td style="text-align: right;">nan %</td></tr>'
    )
