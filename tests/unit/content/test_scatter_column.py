from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, ScatterColumnContentGenerator
from arkas.content.scatter_column import create_template
from arkas.state import ScatterDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


###################################################
#     Tests for ScatterColumnContentGenerator     #
###################################################


def test_scatter_column_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ScatterColumnContentGenerator(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("ScatterColumnContentGenerator(")


def test_scatter_column_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ScatterColumnContentGenerator(ScatterDataFrameState(dataframe, x="col1", y="col2"))
    ).startswith("ScatterColumnContentGenerator(")


def test_scatter_column_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).compute(),
        ContentGenerator,
    )


def test_scatter_column_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ScatterColumnContentGenerator(
        ScatterDataFrameState(dataframe, x="col1", y="col2")
    ).equal(ScatterColumnContentGenerator(ScatterDataFrameState(dataframe, x="col1", y="col2")))


def test_scatter_column_content_generator_equal_false_different_state(
    dataframe: pl.DataFrame,
) -> None:
    assert not ScatterColumnContentGenerator(
        ScatterDataFrameState(dataframe, x="col1", y="col2")
    ).equal(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
        )
    )


def test_scatter_column_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not ScatterColumnContentGenerator(
        ScatterDataFrameState(dataframe, x="col1", y="col2")
    ).equal(42)


def test_scatter_column_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).generate_content(),
        str,
    )


def test_scatter_column_content_generator_generate_content_empty() -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
        ).generate_content(),
        str,
    )


def test_scatter_column_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
        ).generate_content(),
        str,
    )


def test_scatter_column_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).generate_body(),
        str,
    )


def test_scatter_column_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_scatter_column_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).generate_toc(),
        str,
    )


def test_scatter_column_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ScatterColumnContentGenerator(
            ScatterDataFrameState(dataframe, x="col1", y="col2")
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
