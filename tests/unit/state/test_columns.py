from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import TwoColumnDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Int64, "col2": pl.Int32, "col3": pl.Float64},
    )


#############################################
#     Tests for TwoColumnDataFrameState     #
#############################################


def test_two_column_dataframe_state_init_column1_missing(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'missing' is not in the DataFrame:"):
        TwoColumnDataFrameState(dataframe, column1="missing", column2="col3")


def test_two_column_dataframe_state_init_column2_missing(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'missing' is not in the DataFrame:"):
        TwoColumnDataFrameState(dataframe, column1="col1", column2="missing")


def test_two_column_dataframe_state_dataframe(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").dataframe, dataframe
    )


def test_two_column_dataframe_state_column1(dataframe: pl.DataFrame) -> None:
    assert TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").column1 == "col1"


def test_two_column_dataframe_state_column2(dataframe: pl.DataFrame) -> None:
    assert TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").column2 == "col3"


def test_two_column_dataframe_state_nan_policy(dataframe: pl.DataFrame) -> None:
    assert (
        TwoColumnDataFrameState(
            dataframe, column1="col1", column2="col3", nan_policy="raise"
        ).nan_policy
        == "raise"
    )


def test_two_column_dataframe_state_nan_policy_default(dataframe: pl.DataFrame) -> None:
    assert (
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").nan_policy == "propagate"
    )


def test_two_column_dataframe_state_figure_config(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TwoColumnDataFrameState(
            dataframe, column1="col1", column2="col3", figure_config=MatplotlibFigureConfig(dpi=300)
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_two_column_dataframe_state_figure_config_default(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").figure_config,
        MatplotlibFigureConfig(),
    )


def test_two_column_dataframe_state_repr(dataframe: pl.DataFrame) -> None:
    assert repr(TwoColumnDataFrameState(dataframe, column1="col1", column2="col3")).startswith(
        "TwoColumnDataFrameState("
    )


def test_two_column_dataframe_state_str(dataframe: pl.DataFrame) -> None:
    assert str(TwoColumnDataFrameState(dataframe, column1="col1", column2="col3")).startswith(
        "TwoColumnDataFrameState("
    )


def test_two_column_dataframe_state_clone(dataframe: pl.DataFrame) -> None:
    state = TwoColumnDataFrameState(
        dataframe,
        column1="col1",
        column2="col3",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        x=42,
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_two_column_dataframe_state_clone_deep(dataframe: pl.DataFrame) -> None:
    state = TwoColumnDataFrameState(
        dataframe,
        column1="col1",
        column2="col3",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        x=42,
    )
    cloned_state = state.clone()

    assert state.equal(
        TwoColumnDataFrameState(
            dataframe,
            column1="col1",
            column2="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            x=42,
        )
    )
    assert cloned_state.equal(
        TwoColumnDataFrameState(
            dataframe,
            column1="col1",
            column2="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            x=42,
        )
    )
    assert state.dataframe is not cloned_state.dataframe


def test_two_column_dataframe_state_clone_shallow(dataframe: pl.DataFrame) -> None:
    state = TwoColumnDataFrameState(
        dataframe,
        column1="col1",
        column2="col3",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        x=42,
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        TwoColumnDataFrameState(
            dataframe,
            column1="col1",
            column2="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            x=42,
        )
    )
    assert cloned_state.equal(
        TwoColumnDataFrameState(
            dataframe,
            column1="col1",
            column2="col3",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            x=42,
        )
    )
    assert state.dataframe is cloned_state.dataframe


def test_two_column_dataframe_state_equal_true(dataframe: pl.DataFrame) -> None:
    assert TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3")
    )


def test_two_column_dataframe_state_equal_false_different_dataframe(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(
            pl.DataFrame({"col1": [], "col2": [], "col3": []}), column1="col1", column2="col3"
        )
    )


def test_two_column_dataframe_state_equal_false_different_column1(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(dataframe, column1="col2", column2="col3")
    )


def test_two_column_dataframe_state_equal_false_different_column2(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col2")
    )


def test_two_column_dataframe_state_equal_false_different_nan_policy(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3", nan_policy="raise")
    )


def test_two_column_dataframe_state_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(
            dataframe, column1="col1", column2="col3", figure_config=MatplotlibFigureConfig(dpi=300)
        )
    )


def test_two_column_dataframe_state_equal_false_different_kwargs(
    dataframe: pl.DataFrame,
) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3", x=42)
    )


def test_two_column_dataframe_state_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").equal(42)


def test_two_column_dataframe_state_get_arg(dataframe: pl.DataFrame) -> None:
    assert (
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3", x=42).get_arg("x") == 42
    )


def test_two_column_dataframe_state_get_arg_missing(dataframe: pl.DataFrame) -> None:
    assert TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").get_arg("x") is None


def test_two_column_dataframe_state_get_arg_missing_default(dataframe: pl.DataFrame) -> None:
    assert TwoColumnDataFrameState(dataframe, column1="col1", column2="col3").get_arg("x", 42) == 42


def test_two_column_dataframe_state_get_args(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TwoColumnDataFrameState(dataframe, column1="col1", column2="col3", x=42).get_args(),
        {
            "dataframe": dataframe,
            "figure_config": MatplotlibFigureConfig(),
            "column1": "col1",
            "column2": "col3",
            "nan_policy": "propagate",
            "x": 42,
        },
    )
