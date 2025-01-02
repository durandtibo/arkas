from __future__ import annotations

import numpy as np
import pytest

from arkas.content import BalancedAccuracyContentGenerator, ContentGenerator
from arkas.evaluator2 import BalancedAccuracyEvaluator, Evaluator
from arkas.output import BalancedAccuracyOutput
from arkas.plotter import Plotter
from arkas.state import AccuracyState

############################################
#     Tests for BalancedAccuracyOutput     #
############################################


def test_balanced_accuracy_output_repr() -> None:
    assert repr(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("BalancedAccuracyOutput(")


def test_balanced_accuracy_output_str() -> None:
    assert str(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    ).startswith("BalancedAccuracyOutput(")


def test_balanced_accuracy_output_equal_true() -> None:
    assert BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_balanced_accuracy_output_equal_false_different_state() -> None:
    assert not BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
    )


def test_balanced_accuracy_output_equal_false_different_nan_policy() -> None:
    assert not BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 2]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            nan_policy="raise",
        )
    )


def test_balanced_accuracy_output_equal_false_different_type() -> None:
    assert not BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
    ).equal(42)


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_content_generator_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    generator = BalancedAccuracyOutput(state, nan_policy).get_content_generator()
    assert generator.equal(BalancedAccuracyContentGenerator(state, nan_policy))


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_content_generator_lazy_false(nan_policy: str) -> None:
    assert isinstance(
        BalancedAccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            nan_policy=nan_policy,
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_evaluator_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    evaluator = BalancedAccuracyOutput(state, nan_policy).get_evaluator()
    assert evaluator.equal(BalancedAccuracyEvaluator(state, nan_policy))


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_evaluator_lazy_false(nan_policy: str) -> None:
    evaluator = BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy=nan_policy,
    ).get_evaluator(lazy=False)
    assert evaluator.equal(Evaluator({"balanced_accuracy": 1.0, "count": 5}))


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_plotter_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
    )
    plotter = BalancedAccuracyOutput(state, nan_policy).get_plotter()
    assert plotter.equal(Plotter())


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_balanced_accuracy_output_get_plotter_lazy_false(nan_policy: str) -> None:
    plotter = BalancedAccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        ),
        nan_policy=nan_policy,
    ).get_plotter(lazy=False)
    assert plotter.equal(Plotter())
