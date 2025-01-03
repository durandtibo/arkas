from __future__ import annotations

import logging
from typing import Callable

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from arkas.figure import HtmlFigure, MatplotlibFigureConfig
from arkas.figure.base import FigureEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#############################################
#     Tests for FigureEqualityComparator     #
#############################################

FIGURE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=HtmlFigure(),
            expected=HtmlFigure(),
        ),
        id="html figure empty",
    ),
    pytest.param(
        ExamplePair(
            actual=HtmlFigure("meow"),
            expected=HtmlFigure("meow"),
        ),
        id="html figure",
    ),
    pytest.param(
        ExamplePair(
            actual=MatplotlibFigureConfig(),
            expected=MatplotlibFigureConfig(),
        ),
        id="matplotlib figure config",
    ),
]


FIGURE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=42.0,
            expected=HtmlFigure(),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=HtmlFigure(),
            expected=HtmlFigure("meow"),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
    pytest.param(
        ExamplePair(
            actual=MatplotlibFigureConfig(),
            expected=MatplotlibFigureConfig(dpi=50),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_figure_equality_comparator_repr() -> None:
    assert repr(FigureEqualityComparator()) == "FigureEqualityComparator()"


def test_figure_equality_comparator_str() -> None:
    assert str(FigureEqualityComparator()) == "FigureEqualityComparator()"


def test_figure_equality_comparator__eq__true() -> None:
    assert FigureEqualityComparator() == FigureEqualityComparator()


def test_figure_equality_comparator__eq__false() -> None:
    assert FigureEqualityComparator() != 123


def test_figure_equality_comparator_clone() -> None:
    op = FigureEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_figure_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = HtmlFigure()
    assert FigureEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", FIGURE_EQUAL)
def test_figure_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = FigureEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FIGURE_EQUAL)
def test_figure_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = FigureEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FIGURE_NOT_EQUAL)
def test_figure_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = FigureEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", FIGURE_NOT_EQUAL)
def test_figure_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = FigureEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", FIGURE_EQUAL)
@pytest.mark.parametrize("show_difference", [True, False])
def test_objects_are_equal_true(
    function: Callable,
    example: ExamplePair,
    show_difference: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO):
        assert function(example.actual, example.expected, show_difference=show_difference)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", FIGURE_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", FIGURE_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
