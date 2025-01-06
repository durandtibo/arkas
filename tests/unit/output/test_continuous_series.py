from __future__ import annotations

import polars as pl
import pytest

from arkas.content import ContentGenerator, ContinuousSeriesContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import ContinuousSeriesOutput, Output
from arkas.plotter import ContinuousSeriesPlotter, Plotter
from arkas.state import SeriesState


@pytest.fixture
def series() -> pl.Series:
    return pl.Series("col1", [1, 2, 3, 4, 5, 6, 7])


############################################
#     Tests for ContinuousSeriesOutput     #
############################################


def test_continuous_series_output_repr(series: pl.Series) -> None:
    assert repr(ContinuousSeriesOutput(SeriesState(series))).startswith("ContinuousSeriesOutput(")


def test_continuous_series_output_str(series: pl.Series) -> None:
    assert str(ContinuousSeriesOutput(SeriesState(series))).startswith("ContinuousSeriesOutput(")


def test_continuous_series_output_compute(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesOutput(SeriesState(series)).compute(), Output)


def test_continuous_series_output_equal_true(series: pl.Series) -> None:
    assert ContinuousSeriesOutput(SeriesState(series)).equal(
        ContinuousSeriesOutput(SeriesState(series))
    )


def test_continuous_series_output_equal_false_different_state(series: pl.Series) -> None:
    assert not ContinuousSeriesOutput(SeriesState(series)).equal(SeriesState(pl.Series()))


def test_continuous_series_output_equal_false_different_type(series: pl.Series) -> None:
    assert not ContinuousSeriesOutput(SeriesState(series)).equal(42)


def test_continuous_series_output_get_content_generator_lazy_true(series: pl.Series) -> None:
    assert (
        ContinuousSeriesOutput(SeriesState(series))
        .get_content_generator()
        .equal(ContinuousSeriesContentGenerator(SeriesState(series)))
    )


def test_continuous_series_output_get_content_generator_lazy_false(series: pl.Series) -> None:
    assert isinstance(
        ContinuousSeriesOutput(SeriesState(series)).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_continuous_series_output_get_evaluator_lazy_true(series: pl.Series) -> None:
    assert ContinuousSeriesOutput(SeriesState(series)).get_evaluator().equal(Evaluator())


def test_continuous_series_output_get_evaluator_lazy_false(series: pl.Series) -> None:
    assert ContinuousSeriesOutput(SeriesState(series)).get_evaluator(lazy=False).equal(Evaluator())


def test_continuous_series_output_get_plotter_lazy_true(series: pl.Series) -> None:
    assert (
        ContinuousSeriesOutput(SeriesState(series))
        .get_plotter()
        .equal(ContinuousSeriesPlotter(SeriesState(series)))
    )


def test_continuous_series_output_get_plotter_lazy_false(series: pl.Series) -> None:
    assert isinstance(ContinuousSeriesOutput(SeriesState(series)).get_plotter(lazy=False), Plotter)
