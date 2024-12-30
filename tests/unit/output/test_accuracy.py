from __future__ import annotations

import numpy as np
import pytest

from arkas.evaluator2 import AccuracyEvaluator, Evaluator
from arkas.hcg import AccuracyContentGenerator, ContentGenerator
from arkas.output import AccuracyOutput
from arkas.plotter import Plotter
from arkas.state import AccuracyState

####################################
#     Tests for AccuracyOutput     #
####################################


def test_accuracy_output_repr() -> None:
    assert repr(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyOutput(")


def test_accuracy_output_str() -> None:
    assert str(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("AccuracyOutput(")


def test_accuracy_output_equal_true() -> None:
    assert AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_accuracy_output_equal_false_different_state() -> None:
    assert not AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_accuracy_output_equal_false_different_nan_policy() -> None:
    assert not AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            nan_policy="raise",
        )
    )


def test_accuracy_output_equal_false_different_type() -> None:
    assert not AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_content_generator_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    generator = AccuracyOutput(state, nan_policy).get_content_generator()
    assert generator.equal(AccuracyContentGenerator(state, nan_policy))


def test_accuracy_output_get_content_generator_lazy_false() -> None:
    generator = AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).get_content_generator(lazy=False)
    assert isinstance(generator, ContentGenerator)


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_evaluator_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    evaluator = AccuracyOutput(state, nan_policy).get_evaluator()
    assert evaluator.equal(AccuracyEvaluator(state, nan_policy))


def test_accuracy_output_get_evaluator_lazy_false() -> None:
    evaluator = AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).get_evaluator(lazy=False)
    assert evaluator.equal(
        Evaluator(
            {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0}
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_plotter_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    plotter = AccuracyOutput(state, nan_policy).get_plotter()
    assert plotter.equal(Plotter())


def test_accuracy_output_get_plotter_lazy_false() -> None:
    plotter = AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).get_plotter(lazy=False)
    assert plotter.equal(Plotter())
