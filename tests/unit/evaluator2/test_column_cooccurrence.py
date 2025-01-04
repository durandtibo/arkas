from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal

from arkas.evaluator2 import ColumnCooccurrenceEvaluator, Evaluator


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
        }
    )


#################################################
#     Tests for ColumnCooccurrenceEvaluator     #
#################################################


def test_column_cooccurrence_evaluator_repr(dataframe: pl.DataFrame) -> None:
    assert (
        repr(ColumnCooccurrenceEvaluator(dataframe))
        == "ColumnCooccurrenceEvaluator(shape=(7, 3), ignore_self=False)"
    )


def test_column_cooccurrence_evaluator_str(dataframe: pl.DataFrame) -> None:
    assert (
        str(ColumnCooccurrenceEvaluator(dataframe))
        == "ColumnCooccurrenceEvaluator(shape=(7, 3), ignore_self=False)"
    )


def test_column_cooccurrence_evaluator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCooccurrenceEvaluator(dataframe).equal(ColumnCooccurrenceEvaluator(dataframe))


def test_column_cooccurrence_evaluator_equal_false_different_dataframe(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceEvaluator(dataframe).equal(
        ColumnCooccurrenceEvaluator(pl.DataFrame())
    )


def test_column_cooccurrence_evaluator_equal_false_different_ignore_self(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCooccurrenceEvaluator(dataframe).equal(
        ColumnCooccurrenceEvaluator(dataframe, ignore_self=True)
    )


def test_column_cooccurrence_evaluator_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not ColumnCooccurrenceEvaluator(dataframe).equal(42)


def test_column_cooccurrence_evaluator_evaluate(dataframe: pl.DataFrame) -> None:
    evaluator = ColumnCooccurrenceEvaluator(dataframe)
    assert objects_are_equal(
        evaluator.evaluate(),
        {"column_cooccurrence": np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int)},
    )


def test_column_cooccurrence_evaluator_evaluate_empty() -> None:
    evaluator = ColumnCooccurrenceEvaluator(pl.DataFrame())
    assert objects_are_equal(
        evaluator.evaluate(),
        {"column_cooccurrence": np.zeros((0, 0), dtype=int)},
    )


def test_column_cooccurrence_evaluator_evaluate_prefix_suffix(dataframe: pl.DataFrame) -> None:
    evaluator = ColumnCooccurrenceEvaluator(dataframe)
    assert objects_are_equal(
        evaluator.evaluate(prefix="prefix_", suffix="_suffix"),
        {
            "prefix_column_cooccurrence_suffix": np.array(
                [[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int
            )
        },
    )


def test_column_cooccurrence_evaluator_compute(dataframe: pl.DataFrame) -> None:
    assert (
        ColumnCooccurrenceEvaluator(dataframe)
        .compute()
        .equal(
            Evaluator(
                {"column_cooccurrence": np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int)}
            )
        )
    )
