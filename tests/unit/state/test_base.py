from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester

from arkas.state import AccuracyState
from arkas.state.base import StateEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


#############################################
#     Tests for StateEqualityComparator     #
#############################################

STATE_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            expected=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        ),
        id="accuracy state",
    ),
]


STATE_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=42.0,
            expected=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            expected=AccuracyState(
                y_true=np.array([0, 0, 0, 0, 0]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            expected_message="objects are not equal:",
        ),
        id="different elements",
    ),
]


def test_state_equality_comparator_repr() -> None:
    assert repr(StateEqualityComparator()) == "StateEqualityComparator()"


def test_state_equality_comparator_str() -> None:
    assert str(StateEqualityComparator()) == "StateEqualityComparator()"


def test_state_equality_comparator__eq__true() -> None:
    assert StateEqualityComparator() == StateEqualityComparator()


def test_state_equality_comparator__eq__false() -> None:
    assert StateEqualityComparator() != 123


def test_state_equality_comparator_clone() -> None:
    op = StateEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_state_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 1, 0, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    assert StateEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_state_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_EQUAL)
def test_state_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_state_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_state_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = StateEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
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


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", STATE_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
