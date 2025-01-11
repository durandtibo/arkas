from __future__ import annotations

from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import EmptyOutput

#################################
#     Tests for EmptyOutput     #
#################################


def test_empty_output_repr() -> None:
    assert repr(EmptyOutput()) == "EmptyOutput()"


def test_empty_output_str() -> None:
    assert str(EmptyOutput()) == "EmptyOutput()"


def test_empty_output_compute() -> None:
    assert EmptyOutput().compute().equal(EmptyOutput())


def test_empty_output_equal_true() -> None:
    assert EmptyOutput().equal(EmptyOutput())


def test_empty_output_equal_false_different_type() -> None:
    assert not EmptyOutput().equal(42)


def test_empty_output_get_content_generator_lazy_true() -> None:
    assert EmptyOutput().get_content_generator().equal(ContentGenerator())


def test_empty_output_get_content_generator_lazy_false() -> None:
    assert EmptyOutput().get_content_generator(lazy=False).equal(ContentGenerator())


def test_empty_output_get_evaluator_lazy_true() -> None:
    assert EmptyOutput().get_evaluator().equal(Evaluator())


def test_empty_output_get_evaluator_lazy_false() -> None:
    assert EmptyOutput().get_evaluator(lazy=False).equal(Evaluator())
