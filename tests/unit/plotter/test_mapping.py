from __future__ import annotations

from coola import objects_are_equal

from arkas.plotter import MappingPlotter, Plotter

####################################
#     Tests for MappingPlotter     #
####################################


def test_mapping_plotter_repr() -> None:
    assert repr(MappingPlotter({})).startswith("MappingPlotter(")


def test_mapping_plotter_str() -> None:
    assert str(MappingPlotter({})).startswith("MappingPlotter(")


def test_mapping_plotter_equal_true() -> None:
    assert MappingPlotter(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(
        MappingPlotter(
            {
                "one": Plotter(figures={"fig": None}),
                "two": Plotter(),
            }
        )
    )


def test_mapping_plotter_equal_false_different_plotters() -> None:
    assert not MappingPlotter(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(
        MappingPlotter(
            {
                "one": Plotter(figures={"fig": None}),
            }
        )
    )


def test_mapping_plotter_equal_false_different_types() -> None:
    assert not MappingPlotter(
        {
            "one": Plotter(figures={"fig": None}),
            "two": Plotter(),
        }
    ).equal(Plotter(figures={"fig": None}))


def test_mapping_plotter_equal_nan_true() -> None:
    assert MappingPlotter(
        {
            "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
            "two": Plotter(),
        }
    ).equal(
        MappingPlotter(
            {
                "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
                "two": Plotter(),
            }
        ),
        equal_nan=True,
    )


def test_mapping_plotter_equal_nan_false() -> None:
    assert not MappingPlotter(
        {
            "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
            "two": Plotter(),
        }
    ).equal(
        MappingPlotter(
            {
                "one": Plotter(figures={"accuracy": float("nan"), "count": 42}),
                "two": Plotter(),
            }
        ),
    )


def test_mapping_plotter_plot() -> None:
    assert objects_are_equal(
        MappingPlotter(
            {
                "one": Plotter(figures={"fig": None}),
                "two": Plotter(),
            }
        ).plot(),
        {"one": {"fig": None}, "two": {}},
    )


def test_mapping_plotter_plot_empty() -> None:
    assert objects_are_equal(MappingPlotter({}).plot(), {})
