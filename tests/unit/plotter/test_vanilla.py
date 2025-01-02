from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from coola import objects_are_equal

from arkas.plotter import Plotter


@pytest.fixture
def figure() -> plt.Figure:
    return plt.subplots()[0]


#############################
#     Tests for Plotter     #
#############################


def test_plotter_repr() -> None:
    assert repr(Plotter()) == "Plotter(count=0)"


def test_plotter_str() -> None:
    assert str(Plotter()) == "Plotter(count=0)"


def test_plotter_compute(figure: plt.Figure) -> None:
    assert Plotter(figures={"fig": figure}).compute().equal(Plotter(figures={"fig": figure}))


def test_plotter_equal_true() -> None:
    assert Plotter().equal(Plotter())


def test_plotter_equal_false_different_figures(figure: plt.Figure) -> None:
    assert not Plotter().equal(Plotter(figures={"fig": figure}))


def test_plotter_equal_false_different_type() -> None:
    assert not Plotter().equal(42)


def test_plotter_equal_nan_true() -> None:
    assert Plotter(figures={"accuracy": float("nan")}).equal(
        Plotter(figures={"accuracy": float("nan")}),
        equal_nan=True,
    )


def test_plotter_equal_nan_false() -> None:
    assert not Plotter(figures={"accuracy": float("nan")}).equal(
        Plotter(figures={"accuracy": float("nan")})
    )


def test_plotter_plot(figure: plt.Figure) -> None:
    assert objects_are_equal(
        Plotter(figures={"fig": figure}).plot(),
        {"fig": figure},
    )


def test_plotter_plot_empty() -> None:
    assert objects_are_equal(Plotter().plot(), {})


def test_plotter_plot_prefix_suffix(figure: plt.Figure) -> None:
    assert objects_are_equal(
        Plotter(figures={"fig": figure}).plot(prefix="prefix_", suffix="_suffix"),
        {"prefix_fig_suffix": figure},
    )
