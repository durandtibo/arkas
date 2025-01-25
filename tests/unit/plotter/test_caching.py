from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.plotter import BaseCachedPlotter, BaseStateCachedPlotter, Plotter
from arkas.state import AccuracyState, BaseState


class MyCachedPlotter(BaseCachedPlotter):

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return isinstance(other, self.__class__)

    def _plot(self) -> dict:
        return {"metric1": 0.42, "metric2": 1.2}


class MyStateCachedPlotter(BaseStateCachedPlotter):

    def _plot(self) -> dict:
        return {"metric1": 0.42, "metric2": 1.2}


#######################################
#     Tests for BaseCachedPlotter     #
#######################################


def test_base_cached_plotter_compute() -> None:
    assert MyCachedPlotter().compute().equal(Plotter({"metric1": 0.42, "metric2": 1.2}))


def test_base_cached_plotter_plot() -> None:
    plotter = MyCachedPlotter()
    out = plotter.plot()
    assert objects_are_equal(out, {"metric1": 0.42, "metric2": 1.2})
    assert objects_are_equal(plotter._cached_figures, {"metric1": 0.42, "metric2": 1.2})
    assert plotter._cached_figures is not out


def test_base_cached_plotter_plot_multi() -> None:
    plotter = MyCachedPlotter()
    out1 = plotter.plot()
    out2 = plotter.plot()
    assert objects_are_equal(out1, out2)
    assert out1 is not out2


def test_base_cached_plotter_plot_prefix_suffix() -> None:
    plotter = MyCachedPlotter()
    out = plotter.plot(prefix="prefix_", suffix="_suffix")
    assert objects_are_equal(out, {"prefix_metric1_suffix": 0.42, "prefix_metric2_suffix": 1.2})
    assert objects_are_equal(plotter._cached_figures, {"metric1": 0.42, "metric2": 1.2})


############################################
#     Tests for BaseStateCachedPlotter     #
############################################


@pytest.fixture
def state() -> BaseState:
    return AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )


def test_base_state_cached_plotter_repr(state: BaseState) -> None:
    assert repr(MyStateCachedPlotter(state)).startswith("MyStateCachedPlotter(")


def test_base_state_cached_plotter_str(state: BaseState) -> None:
    assert str(MyStateCachedPlotter(state)).startswith("MyStateCachedPlotter(")


def test_base_state_cached_plotter_state(state: BaseState) -> None:
    assert MyStateCachedPlotter(state).state.equal(state)


def test_base_state_cached_plotter_compute(state: BaseState) -> None:
    assert MyStateCachedPlotter(state).compute().equal(Plotter({"metric1": 0.42, "metric2": 1.2}))


def test_base_state_cached_plotter_equal_true(state: BaseState) -> None:
    assert MyStateCachedPlotter(state).equal(MyStateCachedPlotter(state))


def test_base_state_cached_plotter_equal_false_different_state(state: BaseState) -> None:
    assert not MyStateCachedPlotter(state).equal(
        MyStateCachedPlotter(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1, 0]),
                y_pred=np.array([1, 0, 0, 1, 1, 0]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_base_state_cached_plotter_equal_false_different_type(state: BaseState) -> None:
    assert not MyStateCachedPlotter(state).equal(42)


def test_base_state_cached_plotter_plot(state: BaseState) -> None:
    plotter = MyStateCachedPlotter(state)
    out = plotter.plot()
    assert objects_are_equal(out, {"metric1": 0.42, "metric2": 1.2})
    assert objects_are_equal(plotter._cached_figures, {"metric1": 0.42, "metric2": 1.2})
    assert plotter._cached_figures is not out


def test_base_state_cached_plotter_plot_multi(state: BaseState) -> None:
    plotter = MyStateCachedPlotter(state)
    out1 = plotter.plot()
    out2 = plotter.plot()
    assert objects_are_equal(out1, out2)
    assert out1 is not out2


def test_base_state_cached_plotter_plot_prefix_suffix(state: BaseState) -> None:
    plotter = MyStateCachedPlotter(state)
    out = plotter.plot(prefix="prefix_", suffix="_suffix")
    assert objects_are_equal(out, {"prefix_metric1_suffix": 0.42, "prefix_metric2_suffix": 1.2})
    assert objects_are_equal(plotter._cached_figures, {"metric1": 0.42, "metric2": 1.2})
