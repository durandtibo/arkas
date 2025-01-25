from __future__ import annotations

import numpy as np
from coola import objects_are_equal

from arkas.evaluator2 import ColumnCooccurrenceEvaluator, Evaluator
from arkas.state import ColumnCooccurrenceState

#################################################
#     Tests for ColumnCooccurrenceEvaluator     #
#################################################


def test_column_cooccurrence_evaluator_repr() -> None:
    assert repr(
        ColumnCooccurrenceEvaluator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("ColumnCooccurrenceEvaluator(")


def test_column_cooccurrence_evaluator_str() -> None:
    assert str(
        ColumnCooccurrenceEvaluator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("ColumnCooccurrenceEvaluator(")


def test_column_cooccurrence_evaluator_state() -> None:
    assert ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).state.equal(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )


def test_column_cooccurrence_evaluator_equal_true() -> None:
    assert ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        ColumnCooccurrenceEvaluator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_column_cooccurrence_evaluator_equal_false_different_state() -> None:
    assert not ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        ColumnCooccurrenceEvaluator(
            ColumnCooccurrenceState(
                matrix=np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_column_cooccurrence_evaluator_equal_false_different_type() -> None:
    assert not ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(42)


def test_column_cooccurrence_evaluator_evaluate() -> None:
    evaluator = ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"column_cooccurrence": np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int)},
    )


def test_column_cooccurrence_evaluator_evaluate_empty() -> None:
    evaluator = ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(matrix=np.zeros((0, 0)), columns=[])
    )
    assert objects_are_equal(
        evaluator.evaluate(),
        {"column_cooccurrence": np.zeros((0, 0))},
    )


def test_column_cooccurrence_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = ColumnCooccurrenceEvaluator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    )
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_column_cooccurrence_suffix": np.array(
                [[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int
            )
        },
    )


def test_column_cooccurrence_evaluator_compute() -> None:
    assert (
        ColumnCooccurrenceEvaluator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
        .compute()
        .equal(
            Evaluator(
                {"column_cooccurrence": np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int)}
            )
        )
    )
