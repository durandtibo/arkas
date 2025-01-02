from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.plotter import ColumnCooccurrencePlotter, Plotter


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


###############################################
#     Tests for ColumnCooccurrencePlotter     #
###############################################


def test_column_cooccurrence_plotter_repr(dataframe: pl.DataFrame) -> None:
    assert (
        repr(ColumnCooccurrencePlotter(dataframe))
        == "ColumnCooccurrencePlotter(shape=(7, 3), ignore_self=False)"
    )


def test_column_cooccurrence_plotter_str(dataframe: pl.DataFrame) -> None:
    assert (
        str(ColumnCooccurrencePlotter(dataframe))
        == "ColumnCooccurrencePlotter(shape=(7, 3), ignore_self=False)"
    )


def test_column_cooccurrence_plotter_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(ColumnCooccurrencePlotter(dataframe).compute(), Plotter)


def test_column_cooccurrence_plotter_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrencePlotter(dataframe).equal(ColumnCooccurrencePlotter(dataframe))


def test_column_cooccurrence_plotter_equal_false_different_frame(dataframe: pl.DataFrame) -> None:
    assert not ColumnCooccurrencePlotter(dataframe).equal(ColumnCooccurrencePlotter(pl.DataFrame()))


def test_column_cooccurrence_plotter_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ColumnCooccurrencePlotter(dataframe).equal(42)


def test_column_cooccurrence_plotter_equal_nan_true() -> None:
    assert ColumnCooccurrencePlotter(
        pl.DataFrame(
            {
                "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        )
    ).equal(
        ColumnCooccurrencePlotter(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            )
        ),
        equal_nan=True,
    )


def test_column_cooccurrence_plotter_equal_nan_false() -> None:
    assert not ColumnCooccurrencePlotter(
        pl.DataFrame(
            {
                "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        )
    ).equal(
        ColumnCooccurrencePlotter(
            pl.DataFrame(
                {
                    "col1": [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, float("nan")],
                    "col2": [0, 1, 0, 1, 0, 1, 0],
                    "col3": [0, 0, 0, 0, 1, 1, 1],
                }
            )
        )
    )


def test_column_cooccurrence_plotter_plot(dataframe: pl.DataFrame) -> None:
    figures = ColumnCooccurrencePlotter(dataframe).plot()
    assert len(figures) == 1
    assert isinstance(figures["column_cooccurrence"], plt.Figure)


def test_column_cooccurrence_plotter_plot_empty() -> None:
    figures = ColumnCooccurrencePlotter(pl.DataFrame()).plot()
    assert len(figures) == 1
    assert isinstance(figures["column_cooccurrence"], plt.Figure)


def test_column_cooccurrence_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = ColumnCooccurrencePlotter(dataframe).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_column_cooccurrence_suffix"], plt.Figure)


def test_column_cooccurrence_plotter_cooccurrence_matrix(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ColumnCooccurrencePlotter(dataframe).cooccurrence_matrix(),
        np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
    )


def test_column_cooccurrence_plotter_cooccurrence_matrix_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert objects_are_equal(
        ColumnCooccurrencePlotter(dataframe, ignore_self=True).cooccurrence_matrix(),
        np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
    )
