from __future__ import annotations

import numpy as np

from arkas.figure import HtmlFigure, MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plotter import NullValuePlotter, Plotter
from arkas.plotter.null_value import MatplotlibFigureCreator
from arkas.state import NullValueState

######################################
#     Tests for NullValuePlotter     #
######################################


def test_null_value_plotter_repr() -> None:
    assert repr(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValuePlotter(")


def test_null_value_plotter_str() -> None:
    assert str(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValuePlotter(")


def test_null_value_plotter_state() -> None:
    assert NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).state.equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_plotter_compute() -> None:
    assert isinstance(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).compute(),
        Plotter,
    )


def test_null_value_plotter_equal_true() -> None:
    assert NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_null_value_plotter_equal_false_different_state() -> None:
    assert not NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValuePlotter(
            NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[])
        )
    )


def test_null_value_plotter_equal_false_different_type() -> None:
    assert not NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(42)


def test_null_value_plotter_equal_nan_true() -> None:
    assert NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, float("nan")]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, float("nan")]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ),
        equal_nan=True,
    )


def test_null_value_plotter_equal_nan_false() -> None:
    assert not NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, float("nan")]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValuePlotter(
            NullValueState(
                null_count=np.array([1, 2, float("nan")]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_null_value_plotter_plot() -> None:
    figures = NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).plot()
    assert len(figures) == 1
    assert isinstance(figures["null_values"], MatplotlibFigure)


def test_null_value_plotter_plot_empty() -> None:
    figures = NullValuePlotter(
        NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[])
    ).plot()
    assert len(figures) == 1
    assert figures["null_values"].equal(HtmlFigure(MISSING_FIGURE_MESSAGE))


def test_null_value_plotter_plot_prefix_suffix() -> None:
    figures = NullValuePlotter(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).plot(prefix="prefix_", suffix="_suffix")
    assert len(figures) == 1
    assert isinstance(figures["prefix_null_values_suffix"], MatplotlibFigure)


#############################################
#     Tests for MatplotlibFigureCreator     #
#############################################


def test_matplotlib_figure_creator_repr() -> None:
    assert repr(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_str() -> None:
    assert str(MatplotlibFigureCreator()).startswith("MatplotlibFigureCreator(")


def test_matplotlib_figure_creator_create() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_figure_config() -> None:
    assert isinstance(
        MatplotlibFigureCreator().create(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
                figure_config=MatplotlibFigureConfig(yscale="linear", init={}),
            )
        ),
        MatplotlibFigure,
    )


def test_matplotlib_figure_creator_create_empty() -> None:
    assert (
        MatplotlibFigureCreator()
        .create(NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[]))
        .equal(HtmlFigure(MISSING_FIGURE_MESSAGE))
    )
