from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import DataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


####################################
#     Tests for DataFrameState     #
####################################


def test_dataframe_state_dataframe(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(DataFrameState(dataframe).dataframe, dataframe)


def test_dataframe_state_figure_config(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        DataFrameState(dataframe, figure_config=MatplotlibFigureConfig(dpi=300)).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_dataframe_state_figure_config_default(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(DataFrameState(dataframe).figure_config, MatplotlibFigureConfig())


def test_dataframe_state_repr(dataframe: pl.DataFrame) -> None:
    assert repr(DataFrameState(dataframe)).startswith("DataFrameState(")


def test_dataframe_state_str(dataframe: pl.DataFrame) -> None:
    assert str(DataFrameState(dataframe)).startswith("DataFrameState(")


def test_dataframe_state_clone(dataframe: pl.DataFrame) -> None:
    state = DataFrameState(dataframe)
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_dataframe_state_clone_deep(dataframe: pl.DataFrame) -> None:
    state = DataFrameState(dataframe)
    cloned_state = state.clone()

    assert state.equal(DataFrameState(dataframe))
    assert cloned_state.equal(DataFrameState(dataframe))
    assert state.dataframe is not cloned_state.dataframe


def test_dataframe_state_clone_shallow(dataframe: pl.DataFrame) -> None:
    state = DataFrameState(dataframe)
    cloned_state = state.clone(deep=False)

    assert state.equal(DataFrameState(dataframe))
    assert cloned_state.equal(DataFrameState(dataframe))
    assert state.dataframe is cloned_state.dataframe


def test_dataframe_state_equal_true(dataframe: pl.DataFrame) -> None:
    assert DataFrameState(dataframe).equal(DataFrameState(dataframe))


def test_dataframe_state_equal_false_different_dataframe(dataframe: pl.DataFrame) -> None:
    assert not DataFrameState(dataframe).equal(DataFrameState(pl.DataFrame()))


def test_dataframe_state_equal_false_different_figure_config(dataframe: pl.DataFrame) -> None:
    assert not DataFrameState(dataframe).equal(
        DataFrameState(dataframe, figure_config=MatplotlibFigureConfig(dpi=300))
    )


def test_dataframe_state_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not DataFrameState(dataframe).equal(42)


def test_dataframe_state_get_args(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        DataFrameState(dataframe).get_args(),
        {"dataframe": dataframe, "figure_config": MatplotlibFigureConfig()},
    )
