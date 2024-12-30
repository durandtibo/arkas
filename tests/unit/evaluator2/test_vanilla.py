from __future__ import annotations

from coola import objects_are_equal

from arkas.evaluator2 import Evaluator

###############################
#     Tests for Evaluator     #
###############################


def test_evaluator_repr() -> None:
    assert repr(Evaluator()) == "Evaluator(count=0)"


def test_evaluator_str() -> None:
    assert str(Evaluator()) == "Evaluator(count=0)"


def test_evaluator_equal_true() -> None:
    assert Evaluator(metrics={"accuracy": 1.0, "count": 42}).equal(
        Evaluator(metrics={"accuracy": 1.0, "count": 42})
    )


def test_evaluator_equal_false_different_metrics() -> None:
    assert not Evaluator(metrics={"accuracy": 1.0, "count": 42}).equal(
        Evaluator(metrics={"accuracy": 1.0})
    )


def test_evaluator_equal_false_different_type() -> None:
    assert not Evaluator().equal(42)


def test_evaluator_equal_nan_true() -> None:
    assert Evaluator(metrics={"accuracy": float("nan"), "count": 42}).equal(
        Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
        equal_nan=True,
    )


def test_evaluator_equal_nan_false() -> None:
    assert not Evaluator(metrics={"accuracy": float("nan"), "count": 42}).equal(
        Evaluator(metrics={"accuracy": float("nan"), "count": 42})
    )


def test_evaluator_evaluate() -> None:
    assert objects_are_equal(
        Evaluator(metrics={"accuracy": 1.0, "count": 42}).evaluate(),
        {"accuracy": 1.0, "count": 42},
    )


def test_evaluator_evaluate_empty() -> None:
    assert objects_are_equal(Evaluator().evaluate(), {})


def test_evaluator_evaluate_prefix_suffix() -> None:
    assert objects_are_equal(
        Evaluator(metrics={"accuracy": 1.0, "count": 42}).evaluate(
            prefix="prefix_", suffix="_suffix"
        ),
        {"prefix_accuracy_suffix": 1.0, "prefix_count_suffix": 42},
    )
