from __future__ import annotations

from arkas.evaluator2 import Evaluator
from arkas.hcg import ContentGenerator
from arkas.output import Output
from arkas.plotter import Plotter

############################
#     Tests for Output     #
############################


def test_output_repr() -> None:
    assert repr(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
    ).startswith("Output(")


def test_output_str() -> None:
    assert str(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
    ).startswith("Output(")


def test_output_equal_true() -> None:
    assert Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
        plotter=Plotter(),
    ).equal(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
    )


def test_output_equal_false_different_generator() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
        plotter=Plotter(),
    ).equal(
        Output(
            content=ContentGenerator("miaou"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
    )


def test_output_equal_false_different_evaluator() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
        plotter=Plotter(),
    ).equal(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 1.0}),
            plotter=Plotter(),
        )
    )


def test_output_equal_false_different_plotter() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
        plotter=Plotter(),
    ).equal(
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter({"fig": None}),
        )
    )


def test_output_equal_false_different_type() -> None:
    assert not Output(
        content=ContentGenerator("meow"),
        evaluator=Evaluator(metrics={"accuracy": 0.42}),
        plotter=Plotter(),
    ).equal(42)


def test_output_get_content_generator() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
        .get_content_generator()
        .equal(ContentGenerator("meow"))
    )


def test_output_get_evaluator_lazy_true() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
        .get_evaluator()
        .equal(Evaluator(metrics={"accuracy": 0.42}))
    )


def test_output_get_evaluator_lazy_false() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
        .get_evaluator(lazy=False)
        .equal(Evaluator(metrics={"accuracy": 0.42}))
    )


def test_output_get_plotter_lazy_true() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
        .get_plotter()
        .equal(Plotter())
    )


def test_output_get_plotter_lazy_false() -> None:
    assert (
        Output(
            content=ContentGenerator("meow"),
            evaluator=Evaluator(metrics={"accuracy": 0.42}),
            plotter=Plotter(),
        )
        .get_plotter(lazy=False)
        .equal(Plotter())
    )
