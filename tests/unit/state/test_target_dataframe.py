from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import TargetDataFrameState


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


############################################
#     Tests for TargetDataFrameState     #
############################################


def test_target_dataframe_state_init_target_column_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'time' is not in the DataFrame:"):
        TargetDataFrameState(dataframe, target_column="time")


def test_target_dataframe_state_dataframe(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TargetDataFrameState(dataframe, target_column="col3").dataframe, dataframe
    )


def test_target_dataframe_state_target_column(dataframe: pl.DataFrame) -> None:
    assert TargetDataFrameState(dataframe, target_column="col3").target_column == "col3"


def test_target_dataframe_state_figure_config(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TargetDataFrameState(
            dataframe, target_column="col3", figure_config=MatplotlibFigureConfig(dpi=300)
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_target_dataframe_state_figure_config_default(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TargetDataFrameState(dataframe, target_column="col3").figure_config,
        MatplotlibFigureConfig(),
    )


def test_target_dataframe_state_repr(dataframe: pl.DataFrame) -> None:
    assert repr(TargetDataFrameState(dataframe, target_column="col3")).startswith(
        "TargetDataFrameState("
    )


def test_target_dataframe_state_str(dataframe: pl.DataFrame) -> None:
    assert str(TargetDataFrameState(dataframe, target_column="col3")).startswith(
        "TargetDataFrameState("
    )


def test_target_dataframe_state_clone(dataframe: pl.DataFrame) -> None:
    state = TargetDataFrameState(dataframe, target_column="col3")
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_target_dataframe_state_clone_deep(dataframe: pl.DataFrame) -> None:
    state = TargetDataFrameState(
        dataframe,
        target_column="col3",
        figure_config=MatplotlibFigureConfig(dpi=300),
    )
    cloned_state = state.clone()

    assert state.equal(
        TargetDataFrameState(
            dataframe,
            target_column="col3",
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
    assert cloned_state.equal(
        TargetDataFrameState(
            dataframe,
            target_column="col3",
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
    assert state.dataframe is not cloned_state.dataframe


def test_target_dataframe_state_clone_shallow(dataframe: pl.DataFrame) -> None:
    state = TargetDataFrameState(
        dataframe,
        target_column="col3",
        figure_config=MatplotlibFigureConfig(dpi=300),
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        TargetDataFrameState(
            dataframe,
            target_column="col3",
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
    assert cloned_state.equal(
        TargetDataFrameState(
            dataframe,
            target_column="col3",
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
    assert state.dataframe is cloned_state.dataframe


def test_target_dataframe_state_equal_true(dataframe: pl.DataFrame) -> None:
    assert TargetDataFrameState(dataframe, target_column="col3").equal(
        TargetDataFrameState(dataframe, target_column="col3")
    )


def test_target_dataframe_state_equal_false_different_dataframe(dataframe: pl.DataFrame) -> None:
    assert not TargetDataFrameState(dataframe, target_column="col3").equal(
        TargetDataFrameState(pl.DataFrame({"col3": []}), target_column="col3")
    )


def test_target_dataframe_state_equal_false_different_target_column(
    dataframe: pl.DataFrame,
) -> None:
    assert not TargetDataFrameState(dataframe, target_column="col3").equal(
        TargetDataFrameState(dataframe, target_column="col1")
    )


def test_target_dataframe_state_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not TargetDataFrameState(dataframe, target_column="col3").equal(
        TargetDataFrameState(
            dataframe, target_column="col3", figure_config=MatplotlibFigureConfig(dpi=300)
        )
    )


def test_target_dataframe_state_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TargetDataFrameState(dataframe, target_column="col3").equal(42)


def test_target_dataframe_state_get_args(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TargetDataFrameState(dataframe, target_column="col3").get_args(),
        {
            "dataframe": dataframe,
            "figure_config": MatplotlibFigureConfig(),
            "target_column": "col3",
        },
    )
