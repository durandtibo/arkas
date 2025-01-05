from __future__ import annotations

import numpy as np

from arkas.content import ColumnCooccurrenceContentGenerator, ContentGenerator
from arkas.evaluator2 import ColumnCooccurrenceEvaluator, Evaluator
from arkas.output import ColumnCooccurrenceOutput, Output
from arkas.plotter import Plotter
from arkas.state import ColumnCooccurrenceState

##############################################
#     Tests for ColumnCooccurrenceOutput     #
##############################################


def test_column_cooccurrence_output_repr() -> None:
    assert repr(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    ).startswith("ColumnCooccurrenceOutput(")


def test_column_cooccurrence_output_str() -> None:
    assert str(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    ).startswith("ColumnCooccurrenceOutput(")


def test__column_cooccurrence_output_compute() -> None:
    assert isinstance(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ).compute(),
        Output,
    )


def test_column_cooccurrence_output_equal_true() -> None:
    assert ColumnCooccurrenceOutput(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
    )


def test_column_cooccurrence_output_equal_false_different_state() -> None:
    assert not ColumnCooccurrenceOutput(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.zeros((3, 3)), columns=["a", "b", "c"])
        )
    )


def test_column_cooccurrence_output_equal_false_different_type() -> None:
    assert not ColumnCooccurrenceOutput(
        ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ).equal(42)


def test_column_cooccurrence_output_get_content_generator_lazy_true() -> None:
    assert (
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
        .get_content_generator()
        .equal(
            ColumnCooccurrenceContentGenerator(
                ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
            )
        )
    )


def test_column_cooccurrence_output_get_content_generator_lazy_false() -> None:
    assert isinstance(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_column_cooccurrence_output_get_evaluator_lazy_true() -> None:
    assert (
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
        .get_evaluator()
        .equal(
            ColumnCooccurrenceEvaluator(
                ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
            )
        )
    )


def test_column_cooccurrence_output_get_evaluator_lazy_false() -> None:
    assert (
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
        .get_evaluator(lazy=False)
        .equal(Evaluator({"column_cooccurrence": np.ones((3, 3))}))
    )


def test_column_cooccurrence_output_get_plotter_lazy_true() -> None:
    assert (
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        )
        .get_content_generator()
        .equal(
            ColumnCooccurrenceContentGenerator(
                ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
            )
        )
    )


def test_column_cooccurrence_output_get_plotter_lazy_false() -> None:
    assert isinstance(
        ColumnCooccurrenceOutput(
            ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ).get_plotter(lazy=False),
        Plotter,
    )
