from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import SeriesState


@pytest.fixture
def series() -> pl.Series:
    return pl.Series("col1", [1, 2, 3, 4, 5, 6, 7])


#################################
#     Tests for SeriesState     #
#################################


def test_series_state_series(series: pl.Series) -> None:
    assert objects_are_equal(SeriesState(series).series, series)


def test_series_state_figure_config(series: pl.Series) -> None:
    assert objects_are_equal(
        SeriesState(series, figure_config=MatplotlibFigureConfig(dpi=300)).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_series_state_figure_config_default(series: pl.Series) -> None:
    assert objects_are_equal(SeriesState(series).figure_config, MatplotlibFigureConfig())


def test_series_state_repr(series: pl.Series) -> None:
    assert repr(SeriesState(series)).startswith("SeriesState(")


def test_series_state_str(series: pl.Series) -> None:
    assert str(SeriesState(series)).startswith("SeriesState(")


def test_series_state_clone(series: pl.Series) -> None:
    state = SeriesState(series)
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_series_state_clone_deep(series: pl.Series) -> None:
    state = SeriesState(series)
    cloned_state = state.clone()

    assert state.equal(SeriesState(series))
    assert cloned_state.equal(SeriesState(series))
    assert state.series is not cloned_state.series


def test_series_state_clone_shallow(series: pl.Series) -> None:
    state = SeriesState(series)
    cloned_state = state.clone(deep=False)

    assert state.equal(SeriesState(series))
    assert cloned_state.equal(SeriesState(series))
    assert state.series is cloned_state.series


def test_series_state_equal_true(series: pl.Series) -> None:
    assert SeriesState(series).equal(SeriesState(series))


def test_series_state_equal_false_different_series(series: pl.Series) -> None:
    assert not SeriesState(series).equal(SeriesState(pl.Series()))


def test_series_state_equal_false_different_figure_config(series: pl.Series) -> None:
    assert not SeriesState(series).equal(
        SeriesState(series, figure_config=MatplotlibFigureConfig(dpi=300))
    )


def test_series_state_equal_false_different_type(series: pl.Series) -> None:
    assert not SeriesState(series).equal(42)


def test_series_state_get_args(series: pl.Series) -> None:
    assert objects_are_equal(
        SeriesState(series).get_args(),
        {"series": series, "figure_config": MatplotlibFigureConfig()},
    )
