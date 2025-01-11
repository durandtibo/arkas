from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import ColumnCooccurrenceState

#############################################
#     Tests for ColumnCooccurrenceState     #
#############################################


def test_column_cooccurrence_state_init_incorrect_matrix() -> None:
    with pytest.raises(ValueError, match="Incorrect 'matrix'"):
        ColumnCooccurrenceState(
            matrix=np.ones((3, 4)),
            columns=["col1", "col2", "col3"],
        )


def test_column_cooccurrence_state_init_incorrect_columns() -> None:
    with pytest.raises(ValueError, match="The number of columns does not match the matrix shape"):
        ColumnCooccurrenceState(
            matrix=np.ones((3, 3)),
            columns=["col1", "col2", "col3", "col4"],
        )


def test_column_cooccurrence_state_matrix() -> None:
    assert objects_are_equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        ).matrix,
        np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
    )


def test_column_cooccurrence_state_columns() -> None:
    assert objects_are_equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        ).columns,
        ("col1", "col2", "col3"),
    )


def test_column_cooccurrence_state_figure_config() -> None:
    assert objects_are_equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_column_cooccurrence_state_repr() -> None:
    assert repr(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).startswith("ColumnCooccurrenceState(")


def test_column_cooccurrence_state_str() -> None:
    assert str(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).startswith("ColumnCooccurrenceState(")


def test_column_cooccurrence_state_clone() -> None:
    state = ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_column_cooccurrence_state_clone_deep() -> None:
    state = ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone()

    assert state.equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert cloned_state.equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert state.matrix is not cloned_state.matrix


def test_column_cooccurrence_state_clone_shallow() -> None:
    state = ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert cloned_state.equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert state.matrix is cloned_state.matrix


def test_column_cooccurrence_state_equal_true() -> None:
    assert ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )


def test_column_cooccurrence_state_equal_false_different_matrix() -> None:
    assert not ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )


def test_column_cooccurrence_state_equal_false_different_columns() -> None:
    assert not ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["a", "b", "c"],
        )
    )


def test_column_cooccurrence_state_equal_false_different_figure_config() -> None:
    assert not ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )


def test_column_cooccurrence_state_equal_false_different_type() -> None:
    assert not ColumnCooccurrenceState(
        matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
        columns=["col1", "col2", "col3"],
    ).equal(42)


def test_column_cooccurrence_state_from_dataframe() -> None:
    assert ColumnCooccurrenceState.from_dataframe(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        )
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )


def test_column_cooccurrence_state_from_dataframe_ignore_self() -> None:
    assert ColumnCooccurrenceState.from_dataframe(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        ),
        ignore_self=True,
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )


def test_column_cooccurrence_state_from_dataframe_figure_config() -> None:
    assert ColumnCooccurrenceState.from_dataframe(
        pl.DataFrame(
            {
                "col1": [0, 1, 1, 0, 0, 1, 0],
                "col2": [0, 1, 0, 1, 0, 1, 0],
                "col3": [0, 0, 0, 0, 1, 1, 1],
            }
        ),
        figure_config=MatplotlibFigureConfig(dpi=300),
    ).equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )
