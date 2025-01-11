from __future__ import annotations

import numpy as np

from arkas.content import ContentGenerator, NullValueContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import NullValueOutput, Output
from arkas.state import NullValueState

#####################################
#     Tests for NullValueOutput     #
#####################################


def test_null_value_output_repr() -> None:
    assert repr(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValueOutput(")


def test_null_value_output_str() -> None:
    assert str(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValueOutput(")


def test_null_value_output_compute() -> None:
    assert isinstance(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).compute(),
        Output,
    )


def test_null_value_output_equal_true() -> None:
    assert NullValueOutput(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_null_value_output_equal_false_different_state() -> None:
    assert not NullValueOutput(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[]))


def test_null_value_output_equal_false_different_type() -> None:
    assert not NullValueOutput(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(42)


def test_null_value_output_get_content_generator_lazy_true() -> None:
    assert (
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
        .get_content_generator()
        .equal(
            NullValueContentGenerator(
                NullValueState(
                    null_count=np.array([1, 2, 3]),
                    total_count=np.array([7, 7, 7]),
                    columns=["col1", "col2", "col3"],
                )
            )
        )
    )


def test_null_value_output_get_content_generator_lazy_false() -> None:
    assert isinstance(
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


def test_null_value_output_get_evaluator_lazy_true() -> None:
    assert (
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
        .get_evaluator()
        .equal(Evaluator())
    )


def test_null_value_output_get_evaluator_lazy_false() -> None:
    assert (
        NullValueOutput(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
        .get_evaluator(lazy=False)
        .equal(Evaluator())
    )
