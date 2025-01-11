from __future__ import annotations

from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import Output

############################
#     Tests for Output     #
############################


def test_output_repr() -> None:
    assert repr(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
    ).startswith("Output(")


def test_output_str() -> None:
    assert str(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
    ).startswith("Output(")


def test_content_output_compute() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
        .compute()
        .equal(
            Output(
                content=ContentGenerator("meow"),
                evaluator=Evaluator(metrics={"accuracy": 0.42}),
            )
        )
    )


def test_output_equal_true() -> None:
    assert Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
    ).equal(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
    )


def test_output_equal_false_different_generator() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
    ).equal(
        Output(
            content=ContentGenerator("miaou"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
    )


def test_output_equal_false_different_evaluator() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
    ).equal(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 1.0}),
        )
    )


def test_output_equal_false_different_type() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
    ).equal(42)


def test_output_get_content_generator_lazy_true() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
        .get_content_generator()
        .equal(ContentGenerator("meow"))
    )


def test_output_get_content_generator_lazy_false() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
        .get_content_generator(lazy=False)
        .equal(ContentGenerator("meow"))
    )


def test_output_get_evaluator_lazy_true() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
        .get_evaluator()
        .equal(Evaluator(metrics={"accuracy": 0.42}))
    )


def test_output_get_evaluator_lazy_false() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
        )
        .get_evaluator(lazy=False)
        .equal(Evaluator(metrics={"accuracy": 0.42}))
    )
