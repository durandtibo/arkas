from __future__ import annotations

import numpy as np

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import ColumnCooccurrencePlotter, Plotter
from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
from arkas.state import ColumnCooccurrenceState

###############################################
#     Tests for ColumnCooccurrencePlotter     #
###############################################


def test_column_cooccurrence_plotter_repr() -> None:
    assert repr(
        ColumnCooccurrencePlotter(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    ).startswith("ColumnCooccurrencePlotter(")


def test_column_cooccurrence_plotter_str() -> None:
    assert str(
        ColumnCooccurrencePlotter(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    ).startswith("ColumnCooccurrencePlotter(")


def test_column_cooccurrence_plotter_compute() -> None:
    assert isinstance(
        ColumnCooccurrencePlotter(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ).compute(),
        Plotter,
    )


def test_column_cooccurrence_plotter_equal_true() -> None:
    assert ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(
        ColumnCooccurrencePlotter(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    )


def test_column_cooccurrence_plotter_equal_false_different_state() -> None:
    assert not ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(
        ColumnCooccurrencePlotter(
            ColumnCooccurrenceState(matrix=np.zeros((3, 3)), columns=["a", "b", "c"])
        )
    )


def test_column_cooccurrence_plotter_equal_false_different_type() -> None:
    assert not ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(42)


def test_column_cooccurrence_plotter_plot() -> None:
    figures = ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).plot()
    assert len(figures) == 1
    assert isinstance(figures["column_cooccurrence"], MatplotlibFigure)


def test_column_cooccurrence_plotter_plot_empty() -> None:
    figures = ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.zeros((0, 0)), columns=[])
    ).plot()
    assert len(figures) == 1
    assert figures["column_cooccurrence"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_column_cooccurrence_plotter_plot_prefix_suffix() -> None:
    figures = ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_column_cooccurrence_suffix"], MatplotlibFigure)


def test_column_cooccurrence_plotter_plot_figure_config() -> None:
    figures = ColumnCooccurrencePlotter(
        ColumnCooccurrenceState(
            matrix=np.ones((3, 3)),
            columns=["a", "b", "c"],
            figure_config=MatplotlibFigureConfig(dpi=50),
        )
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
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_large() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            ColumnCooccurrenceState(matrix=np.ones((50, 50)), columns=list(map(str, range(50))))
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(ColumnCooccurrenceState(matrix=np.ones((0, 0)), columns=[]))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
