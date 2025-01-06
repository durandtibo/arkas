from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import NullValueState

####################################
#     Tests for NullValueState     #
####################################


def test_null_value_state_init_incorrect_null_count() -> None:
    with pytest.raises(ValueError, match=r"'columns' \(3\) and 'null_count' \(4\) do not match"):
        NullValueState(
            null_count=np.array([1, 2, 3, 4]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )


def test_null_value_state_init_incorrect_total_count() -> None:
    with pytest.raises(ValueError, match=r"'columns' \(3\) and 'total_count' \(4\) do not match"):
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )


def test_null_value_state_null_count() -> None:
    assert objects_are_equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        ).null_count,
        np.array([1, 2, 3]),
    )


def test_null_value_state_total_count() -> None:
    assert objects_are_equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        ).total_count,
        np.array([7, 7, 7]),
    )


def test_null_value_state_columns() -> None:
    assert objects_are_equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        ).columns,
        ("col1", "col2", "col3"),
    )


def test_null_value_state_figure_config() -> None:
    assert objects_are_equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_null_value_state_repr() -> None:
    assert repr(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).startswith("NullValueState(")


def test_null_value_state_str() -> None:
    assert str(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).startswith("NullValueState(")


def test_null_value_state_clone() -> None:
    state = NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_null_value_state_clone_deep() -> None:
    state = NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone()

    assert state.equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )
    assert cloned_state.equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )
    assert state.null_count is not cloned_state.null_count
    assert state.total_count is not cloned_state.total_count


def test_null_value_state_clone_shallow() -> None:
    state = NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )
    assert cloned_state.equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_equal_true() -> None:
    assert NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_equal_false_different_null_count() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([0, 1, 2]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_equal_false_different_total_count() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([8, 8, 8]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_equal_false_different_columns() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["a", "b", "c"],
        )
    )


def test_null_value_state_equal_false_different_figure_config() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )


def test_null_value_state_equal_false_different_type() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, 3]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(42)


def test_null_value_state_equal_nan_true() -> None:
    assert NullValueState(
        null_count=np.array([1, 2, float("nan")]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, float("nan")]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        ),
        equal_nan=True,
    )


def test_null_value_state_equal_nan_false() -> None:
    assert not NullValueState(
        null_count=np.array([1, 2, float("nan")]),
        total_count=np.array([7, 7, 7]),
        columns=["col1", "col2", "col3"],
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, float("nan")]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_from_dataframe() -> None:
    assert NullValueState.from_dataframe(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, None],
                "col2": [0, 1, None, None, 0, 1, 0],
                "col3": [None, 0, 0, 0, None, 1, None],
            }
        )
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    )


def test_null_value_state_from_dataframe_figure_config() -> None:
    assert NullValueState.from_dataframe(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, None],
                "col2": [0, 1, None, None, 0, 1, 0],
                "col3": [None, 0, 0, 0, None, 1, None],
            }
        ),
        figure_config=MatplotlibFigureConfig(dpi=300),
    ).equal(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
