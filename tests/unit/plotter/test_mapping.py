from __future__ import annotations

from coola import objects_are_equal

from arkas.plotter import Plotter, PlotterDict

#################################
#     Tests for PlotterDict     #
#################################


def test_plotter_dict_repr() -> None:
    assert repr(PlotterDict({})).startswith("PlotterDict(")


def test_plotter_dict_str() -> None:
    assert str(PlotterDict({})).startswith("PlotterDict(")


def test_plotter_dict_compute() -> None:
    assert (
        PlotterDict(
            {
                "one": Plotter(figures={"fig": None}),
                "two": Plotter(),
            }
        )
        .compute()
        .equal(
            Plotter(
                {
                    "one": {"fig": None},
                    "two": {},
                }
            )
        )
    )


def test_plotter_dict_equal_true() -> None:
    assert PlotterDict(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(
        PlotterDict(
            {
                "one": Plotter(figures={"fig": None}),
                "two": Plotter(),
            }
        )
    )


def test_plotter_dict_equal_false_different_plotters() -> None:
    assert not PlotterDict(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(PlotterDict({"one": Plotter(figures={"fig": None})}))


def test_plotter_dict_equal_false_different_types() -> None:
    assert not PlotterDict(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(Plotter(figures={"fig": None}))


def test_plotter_dict_equal_nan_true() -> None:
    assert PlotterDict(
        {
            "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
            "two": Plotter(),
        }
    ).equal(
        PlotterDict(
            {
                "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
                "two": Plotter(),
            }
        ),
        equal_nan=True,
    )


def test_plotter_dict_equal_nan_false() -> None:
    assert not PlotterDict(
        {
            "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
            "two": Plotter(),
        }
    ).equal(
        PlotterDict(
            {
                "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
                "two": Plotter(),
            }
        ),
    )


def test_plotter_dict_plot() -> None:
    assert objects_are_equal(
        PlotterDict(
            {
                "one": Plotter(figures={"fig": None}),
                "two": Plotter(),
            }
        ).plot(),
        {"one": {"fig": None}, "two": {}},
    )


def test_plotter_dict_plot_empty() -> None:
    assert objects_are_equal(PlotterDict({}).plot(), {})
