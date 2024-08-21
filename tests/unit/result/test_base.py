from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pytest
from coola import objects_are_allclose, objects_are_equal
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from arkas.result import AccuracyResult, Result
from arkas.result.base import ResultEqualityComparator
from tests.unit.helpers import ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


##############################################
#     Tests for ResultEqualityComparator     #
##############################################

STATE_FUNCTIONS = [objects_are_equal, objects_are_allclose]

STATE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Result(metrics={"accuracy": 1.0, "count": 42}),
            expected=Result(metrics={"accuracy": 1.0, "count": 42}),
        ),
        id="result",
    ),
    pytest.param(
        ExamplePair(
            actual=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            ),
            expected=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            ),
        ),
        id="accuracy result",
    ),
]


STATE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=Result(metrics={"accuracy": 1.0, "count": 42}),
            expected=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            ),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=Result(metrics={"accuracy": 1.0, "count": 42}),
            expected=Result(metrics={"accuracy": 0.6, "count": 42}),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_result_equality_comparator_repr() -> None:
    assert repr(ResultEqualityComparator()) == "ResultEqualityComparator()"


def test_result_equality_comparator_str() -> None:
    assert str(ResultEqualityComparator()) == "ResultEqualityComparator()"


def test_result_equality_comparator__eq__true() -> None:
    assert ResultEqualityComparator() == ResultEqualityComparator()


def test_result_equality_comparator__eq__false() -> None:
    assert ResultEqualityComparator() != 123


def test_result_equality_comparator_clone() -> None:
    op = ResultEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_result_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = Result(metrics={"accuracy": 1.0, "count": 42})
    assert ResultEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_result_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ResultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_result_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ResultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_result_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ResultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_result_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ResultEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_EQUAL)
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


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", STATE_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
