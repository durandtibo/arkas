from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import ScatterDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        },
        schema={"col1": pl.Int64, "col2": pl.Int64, "col3": pl.Int64},
    )


###########################################
#     Tests for ScatterDataFrameState     #
###########################################


def test_scatter_dataframe_state_init_missing_x(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'col' is not in the DataFrame:"):
        ScatterDataFrameState(dataframe, x="col", y="col2")


def test_scatter_dataframe_state_init_missing_y(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'col' is not in the DataFrame:"):
        ScatterDataFrameState(dataframe, x="col1", y="col")


def test_scatter_dataframe_state_init_missing_color(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'col' is not in the DataFrame:"):
        ScatterDataFrameState(dataframe, x="col1", y="col2", color="col")


def test_scatter_dataframe_state_dataframe(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2").dataframe, dataframe
    )


def test_scatter_dataframe_state_x(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").x == "col1"


def test_scatter_dataframe_state_y(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").y == "col2"


def test_scatter_dataframe_state_color(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2", color="col3").color == "col3"


def test_scatter_dataframe_state_color_default(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").color is None


def test_scatter_dataframe_state_nan_policy(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterDataFrameState(dataframe, x="col1", y="col2", nan_policy="raise").nan_policy
        == "raise"
    )


def test_scatter_dataframe_state_nan_policy_default(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").nan_policy == "propagate"


def test_scatter_dataframe_state_figure_config(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ScatterDataFrameState(
            dataframe, x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=300)
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_scatter_dataframe_state_figure_config_default(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2").figure_config,
        MatplotlibFigureConfig(),
    )


def test_scatter_dataframe_state_repr(dataframe: pl.DataFrame) -> None:
    assert repr(ScatterDataFrameState(dataframe, x="col1", y="col2")).startswith(
        "ScatterDataFrameState("
    )


def test_scatter_dataframe_state_str(dataframe: pl.DataFrame) -> None:
    assert str(ScatterDataFrameState(dataframe, x="col1", y="col2")).startswith(
        "ScatterDataFrameState("
    )


def test_scatter_dataframe_state_clone(dataframe: pl.DataFrame) -> None:
    state = ScatterDataFrameState(
        dataframe,
        x="col1",
        y="col2",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_scatter_dataframe_state_clone_deep(dataframe: pl.DataFrame) -> None:
    state = ScatterDataFrameState(
        dataframe,
        x="col1",
        y="col2",
        color="col3",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone()

    assert state.equal(
        ScatterDataFrameState(
            dataframe,
            x="col1",
            y="col2",
            color="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert cloned_state.equal(
        ScatterDataFrameState(
            dataframe,
            x="col1",
            y="col2",
            color="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert state.dataframe is not cloned_state.dataframe


def test_scatter_dataframe_state_clone_shallow(dataframe: pl.DataFrame) -> None:
    state = ScatterDataFrameState(
        dataframe,
        x="col1",
        y="col2",
        color="col3",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        ScatterDataFrameState(
            dataframe,
            x="col1",
            y="col2",
            color="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert cloned_state.equal(
        ScatterDataFrameState(
            dataframe,
            x="col1",
            y="col2",
            color="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert state.dataframe is cloned_state.dataframe


def test_scatter_dataframe_state_equal_true(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2")
    )


def test_scatter_dataframe_state_equal_false_different_dataframe(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(pl.DataFrame({"col1": [], "col2": []}), x="col1", y="col2")
    )


def test_scatter_dataframe_state_equal_false_different_x(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(dataframe, x="col3", y="col2")
    )


def test_scatter_dataframe_state_equal_false_different_y(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(dataframe, x="col1", y="col3")
    )


def test_scatter_dataframe_state_equal_false_different_color(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2", color="col3").equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2", color="col1")
    )


def test_scatter_dataframe_state_equal_false_different_nan_policy(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2", nan_policy="raise")
    )


def test_scatter_dataframe_state_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(
            dataframe, x="col1", y="col2", figure_config=MatplotlibFigureConfig(dpi=300)
        )
    )


def test_scatter_dataframe_state_equal_false_different_kwargs(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2", column="col")
    )


def test_scatter_dataframe_state_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ScatterDataFrameState(dataframe, x="col1", y="col2").equal(42)


def test_scatter_dataframe_state_get_arg(dataframe: pl.DataFrame) -> None:
    assert (
        ScatterDataFrameState(dataframe, x="col1", y="col2", column="col").get_arg("column")
        == "col"
    )


def test_scatter_dataframe_state_get_arg_missing(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").get_arg("x") is None


def test_scatter_dataframe_state_get_arg_missing_default(dataframe: pl.DataFrame) -> None:
    assert ScatterDataFrameState(dataframe, x="col1", y="col2").get_arg("x", 42) == 42


def test_scatter_dataframe_state_get_args(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        ScatterDataFrameState(dataframe, x="col1", y="col2", color="col3", column="col").get_args(),
        {
            "dataframe": dataframe,
            "figure_config": MatplotlibFigureConfig(),
            "x": "col1",
            "y": "col2",
            "color": "col3",
            "nan_policy": "propagate",
            "column": "col",
        },
    )
