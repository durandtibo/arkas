from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING, Callable

import pytest
from coola.equality import EqualityConfig
from coola.equality.testers import EqualityTester
from objectory import OBJECT_TARGET

from arkas.exporter import MetricExporter, is_exporter_config, setup_exporter
from arkas.exporter.base import ExporterEqualityComparator
from tests.unit.helpers import COMPARATOR_FUNCTIONS, ExamplePair

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config() -> EqualityConfig:
    return EqualityConfig(tester=EqualityTester())


########################################
#     Tests for is_exporter_config     #
########################################


def test_is_exporter_config_true() -> None:
    assert is_exporter_config(
        {
            OBJECT_TARGET: "arkas.exporter.MetricExporter",
            "path": "/path/to/data.csv",
        }
    )


def test_is_exporter_config_false() -> None:
    assert not is_exporter_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_exporter     #
####################################


def test_setup_exporter_object(tmp_path: Path) -> None:
    exporter = MetricExporter(
        path=tmp_path.joinpath("report.html"),
    )
    assert setup_exporter(exporter) is exporter


def test_setup_exporter_dict() -> None:
    assert isinstance(
        setup_exporter(
            {
                OBJECT_TARGET: "arkas.exporter.MetricExporter",
                "path": "/path/to/data.csv",
            }
        ),
        MetricExporter,
    )


def test_setup_exporter_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_exporter({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages


################################################
#     Tests for ExporterEqualityComparator     #
################################################


EVALUATOR_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=MetricExporter("/data/metrics.pkl"),
            expected=MetricExporter("/data/metrics.pkl"),
        ),
        id="exporter",
    ),
]

EVALUATOR_NOT_EQUAL = [
    pytest.param(
        ExamplePair(
            actual=MetricExporter("/data/metrics.pkl"),
            expected=42,
            expected_message="objects have different types:",
        ),
        id="different types",
    ),
    pytest.param(
        ExamplePair(
            actual=MetricExporter("/data/metrics.pkl"),
            expected=MetricExporter("/data/my_metrics.pkl"),
            expected_message="objects are not equal:",
        ),
        id="different values",
    ),
]


def test_exporter_equality_comparator_repr() -> None:
    assert repr(ExporterEqualityComparator()) == "ExporterEqualityComparator()"


def test_exporter_equality_comparator_str() -> None:
    assert str(ExporterEqualityComparator()) == "ExporterEqualityComparator()"


def test_exporter_equality_comparator__eq__true() -> None:
    assert ExporterEqualityComparator() == ExporterEqualityComparator()


def test_exporter_equality_comparator__eq__false() -> None:
    assert ExporterEqualityComparator() != 123


def test_exporter_equality_comparator_clone() -> None:
    op = ExporterEqualityComparator()
    op_cloned = op.clone()
    assert op is not op_cloned
    assert op == op_cloned


def test_exporter_equality_comparator_equal_true_same_object(config: EqualityConfig) -> None:
    x = MetricExporter("/data/metrics.pkl")
    assert ExporterEqualityComparator().equal(x, x, config)


@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
def test_exporter_equality_comparator_equal_true(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
def test_exporter_equality_comparator_equal_true_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_exporter_equality_comparator_equal_false(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert not caplog.messages


@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_exporter_equality_comparator_equal_false_show_difference(
    example: ExamplePair,
    config: EqualityConfig,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config.show_difference = True
    comparator = ExporterEqualityComparator()
    with caplog.at_level(logging.INFO):
        assert not comparator.equal(actual=example.actual, expected=example.expected, config=config)
        assert caplog.messages[-1].startswith(example.expected_message)


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EVALUATOR_EQUAL)
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
@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_objects_are_equal_false(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected)
        assert not caplog.messages


@pytest.mark.parametrize("function", COMPARATOR_FUNCTIONS)
@pytest.mark.parametrize("example", EVALUATOR_NOT_EQUAL)
def test_objects_are_equal_false_show_difference(
    function: Callable, example: ExamplePair, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO):
        assert not function(example.actual, example.expected, show_difference=True)
        assert caplog.messages[-1].startswith(example.expected_message)
