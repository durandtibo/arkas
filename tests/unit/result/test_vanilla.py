from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from coola import objects_are_equal

from arkas.result import Result


@pytest.fixture
def figure() -> plt.Figure:
    return plt.subplots()[0]


############################
#     Tests for Result     #
############################


def test_result_repr() -> None:
    assert repr(Result()) == "Result(metrics=0, figures=0)"


def test_result_str() -> None:
    assert str(Result()) == "Result(metrics=0, figures=0)"


def test_result_compute_metrics() -> None:
    assert objects_are_equal(
        Result(metrics={"accuracy": 1.0, "count": 42}).compute_metrics(),
        {"accuracy": 1.0, "count": 42},
    )


def test_result_compute_metrics_empty() -> None:
    assert objects_are_equal(Result().compute_metrics(), {})


def test_result_compute_metrics_prefix_suffix() -> None:
    assert objects_are_equal(
        Result(metrics={"accuracy": 1.0, "count": 42}).compute_metrics(
            prefix="prefix_", suffix="_suffix"
        ),
        {"prefix_accuracy_suffix": 1.0, "prefix_count_suffix": 42},
    )


def test_result_generate_figures(figure: plt.Figure) -> None:
    assert objects_are_equal(
        Result(figures={"accuracy": figure}).generate_figures(),
        {"accuracy": figure},
    )


def test_result_generate_figures_empty() -> None:
    assert objects_are_equal(Result().generate_figures(), {})


def test_result_generate_figures_prefix_suffix(figure: plt.Figure) -> None:
    assert objects_are_equal(
        Result(figures={"accuracy": figure}).generate_figures(prefix="prefix_", suffix="_suffix"),
        {"prefix_accuracy_suffix": figure},
    )
