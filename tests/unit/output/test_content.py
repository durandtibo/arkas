from __future__ import annotations

from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import ContentOutput
from arkas.plotter import Plotter

###################################
#     Tests for ContentOutput     #
###################################


def test_content_output_repr() -> None:
    assert repr(ContentOutput("meow")).startswith("ContentOutput(")


def test_content_output_str() -> None:
    assert str(ContentOutput("meow")).startswith("ContentOutput(")


def test_content_output_compute() -> None:
    assert ContentOutput("meow").compute().equal(ContentOutput("meow"))


def test_content_output_equal_true() -> None:
    assert ContentOutput("meow").equal(ContentOutput("meow"))


def test_content_output_equal_false_different_type() -> None:
    assert not ContentOutput("meow").equal(42)


def test_content_output_get_content_generator_lazy_true() -> None:
    assert ContentOutput("meow").get_content_generator().equal(ContentGenerator("meow"))


def test_content_output_get_content_generator_lazy_false() -> None:
    assert ContentOutput("meow").get_content_generator(lazy=False).equal(ContentGenerator("meow"))


def test_content_output_get_evaluator_lazy_true() -> None:
    assert ContentOutput("meow").get_evaluator().equal(Evaluator())


def test_content_output_get_evaluator_lazy_false() -> None:
    assert ContentOutput("meow").get_evaluator(lazy=False).equal(Evaluator())


def test_content_output_get_plotter_lazy_true() -> None:
    assert ContentOutput("meow").get_plotter().equal(Plotter())


def test_content_output_get_plotter_lazy_false() -> None:
    assert ContentOutput("meow").get_plotter(lazy=False).equal(Plotter())
