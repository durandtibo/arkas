from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import ColumnCooccurrencePlotter, Plotter
from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator


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


def test_column_cooccurrence_plotter_equal_false_different_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrencePlotter(dataframe).equal(
        ColumnCooccurrencePlotter(dataframe, ignore_self=True)
    )


def test_column_cooccurrence_plotter_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrencePlotter(dataframe).equal(
        ColumnCooccurrencePlotter(dataframe, figure_config=MatplotlibFigureConfig(dpi=100))
    )


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
    assert isinstance(figures["column_cooccurrence"], MatplotlibFigure)


def test_column_cooccurrence_plotter_plot_empty() -> None:
    figures = ColumnCooccurrencePlotter(pl.DataFrame()).plot()
    assert len(figures) == 1
    assert figures["column_cooccurrence"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_column_cooccurrence_plotter_plot_prefix_suffix(dataframe: pl.DataFrame) -> None:
    figures = ColumnCooccurrencePlotter(dataframe).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_column_cooccurrence_suffix"], MatplotlibFigure)


def test_column_cooccurrence_plotter_compute_cooccurrence_matrix(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ColumnCooccurrencePlotter(dataframe).compute_cooccurrence_matrix(),
        np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
    )


def test_column_cooccurrence_plotter_compute_cooccurrence_matrix_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert objects_are_equal(
        ColumnCooccurrencePlotter(dataframe, ignore_self=True).compute_cooccurrence_matrix(),
        np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
    )


def test_column_cooccurrence_plotter_plot_figure_config(dataframe: pl.DataFrame) -> None:
    figures = ColumnCooccurrencePlotter(
        dataframe, figure_config=MatplotlibFigureConfig(dpi=50)
    ).plot()
    assert len(figures) == 1
    assert isinstance(figures["column_cooccurrence"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create_small() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            matrix=np.ones((3, 3)), columns=["a", "b", "c"], config=MatplotlibFigureConfig()
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_large() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            matrix=np.ones((50, 50)),
            columns=list(map(str, range(50))),
            config=MatplotlibFigureConfig(),
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(matrix=np.ones((0, 0)), columns=[], config=MatplotlibFigureConfig())
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
