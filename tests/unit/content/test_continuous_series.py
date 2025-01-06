from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, ContinuousSeriesContentGenerator
from arkas.content.continuous_series import create_template
from arkas.state import SeriesState


@pytest.fixture
def series() -> pl.Series:
    return pl.Series("col1", [1, 2, 3, 4, 5, 6, 7])


######################################################
#     Tests for ContinuousSeriesContentGenerator     #
######################################################


def test_continuous_series_content_generator_repr(series: pl.Series) -> None:
    assert repr(ContinuousSeriesContentGenerator(SeriesState(series))).startswith(
        "ContinuousSeriesContentGenerator("
    )


def test_continuous_series_content_generator_str(series: pl.Series) -> None:
    assert str(ContinuousSeriesContentGenerator(SeriesState(series))).startswith(
        "ContinuousSeriesContentGenerator("
    )


def test_continuous_series_content_generator_compute(series: pl.Series) -> None:
    assert isinstance(
        ContinuousSeriesContentGenerator(SeriesState(series)).compute(), ContentGenerator
    )


def test_continuous_series_content_generator_equal_true(series: pl.Series) -> None:
    assert ContinuousSeriesContentGenerator(SeriesState(series)).equal(
        ContinuousSeriesContentGenerator(SeriesState(series))
    )


def test_continuous_series_content_generator_equal_false_different_state(
    series: pl.Series,
) -> None:
    assert not ContinuousSeriesContentGenerator(SeriesState(series)).equal(
        ContinuousSeriesContentGenerator(SeriesState(pl.Series()))
    )


def test_continuous_series_content_generator_equal_false_different_type(
    series: pl.Series,
) -> None:
    assert not ContinuousSeriesContentGenerator(SeriesState(series)).equal(42)


def test_continuous_series_content_generator_generate_content(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesContentGenerator(SeriesState(series)).generate_content(), str)


def test_continuous_series_content_generator_generate_content_empty() -> None:
    assert isinstance(
        ContinuousSeriesContentGenerator(SeriesState(pl.Series())).generate_content(), str
    )


def test_continuous_series_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        ContinuousSeriesContentGenerator(SeriesState(pl.Series([]))).generate_content(),
        str,
    )


def test_continuous_series_content_generator_generate_body(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesContentGenerator(SeriesState(series)).generate_body(), str)


def test_continuous_series_content_generator_generate_body_args(series: pl.Series) -> None:
    assert isinstance(
        ContinuousSeriesContentGenerator(SeriesState(series)).generate_body(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


def test_continuous_series_content_generator_generate_toc(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesContentGenerator(SeriesState(series)).generate_toc(), str)


def test_continuous_series_content_generator_generate_toc_args(series: pl.Series) -> None:
    assert isinstance(
        ContinuousSeriesContentGenerator(SeriesState(series)).generate_toc(
            number="1.", tags=["meow"], depth=1
        ),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
