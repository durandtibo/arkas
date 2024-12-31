from __future__ import annotations

from coola import objects_are_equal

from arkas.evaluator2 import Evaluator, EvaluatorDict

######################################
#     Tests for EvaluatorDict     #
######################################


def test_evaluator_dict_repr() -> None:
    assert repr(EvaluatorDict({})).startswith("EvaluatorDict(")


def test_evaluator_dict_str() -> None:
    assert str(EvaluatorDict({})).startswith("EvaluatorDict(")


def test_evaluator_dict_equal_true() -> None:
    assert EvaluatorDict(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        EvaluatorDict(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        )
    )


def test_evaluator_dict_equal_false_different_evaluators() -> None:
    assert not EvaluatorDict(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        EvaluatorDict(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            }
        )
    )


def test_evaluator_dict_equal_false_different_types() -> None:
    assert not EvaluatorDict(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(Evaluator(metrics={"accuracy": 62.0, "count": 42}))


def test_evaluator_dict_equal_nan_true() -> None:
    assert EvaluatorDict(
        {
            "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        EvaluatorDict(
            {
                "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
        equal_nan=True,
    )


def test_evaluator_dict_equal_nan_false() -> None:
    assert not EvaluatorDict(
        {
            "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        EvaluatorDict(
            {
                "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
    )


def test_evaluator_dict_evaluate() -> None:
    assert objects_are_equal(
        EvaluatorDict(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ).evaluate(),
        {"one": {"accuracy": 62.0, "count": 42}, "two": {"accuracy": 42.0, "count": 30}},
    )


def test_evaluator_dict_evaluate_empty() -> None:
    assert objects_are_equal(EvaluatorDict({}).evaluate(), {})
