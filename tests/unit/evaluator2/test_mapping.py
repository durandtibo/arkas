from __future__ import annotations

from coola import objects_are_equal

from arkas.evaluator2 import Evaluator, MappingEvaluator

######################################
#     Tests for MappingEvaluator     #
######################################


def test_mapping_evaluator_repr() -> None:
    assert repr(MappingEvaluator({})).startswith("MappingEvaluator(")


def test_mapping_evaluator_str() -> None:
    assert str(MappingEvaluator({})).startswith("MappingEvaluator(")


def test_mapping_evaluator_equal_true() -> None:
    assert MappingEvaluator(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingEvaluator(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        )
    )


def test_mapping_evaluator_equal_false_different_evaluators() -> None:
    assert not MappingEvaluator(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingEvaluator(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            }
        )
    )


def test_mapping_evaluator_equal_false_different_types() -> None:
    assert not MappingEvaluator(
        {
            "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(Evaluator(metrics={"accuracy": 62.0, "count": 42}))


def test_mapping_evaluator_equal_nan_true() -> None:
    assert MappingEvaluator(
        {
            "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingEvaluator(
            {
                "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
        equal_nan=True,
    )


def test_mapping_evaluator_equal_nan_false() -> None:
    assert not MappingEvaluator(
        {
            "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
            "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingEvaluator(
            {
                "one": Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
    )


def test_mapping_evaluator_evaluate() -> None:
    assert objects_are_equal(
        MappingEvaluator(
            {
                "one": Evaluator(metrics={"accuracy": 62.0, "count": 42}),
                "two": Evaluator(metrics={"accuracy": 42.0, "count": 30}),
            }
        ).evaluate(),
        {"one": {"accuracy": 62.0, "count": 42}, "two": {"accuracy": 42.0, "count": 30}},
    )


def test_mapping_evaluator_evaluate_empty() -> None:
    assert objects_are_equal(MappingEvaluator({}).evaluate(), {})
