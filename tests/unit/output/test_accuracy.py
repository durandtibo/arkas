from __future__ import annotations

import numpy as np
import pytest

from arkas.content import AccuracyContentGenerator, ContentGenerator
from arkas.evaluator2 import AccuracyEvaluator, Evaluator
from arkas.output import AccuracyOutput, Output
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


def test_accuracy_output_compute() -> None:
    assert (
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
        )
        .compute()
        .equal(
            Output(
                content=ContentGenerator(
                    "<ul>\n"
                    "  <li><b>accuracy</b>: 1.0000 (5/5)</li>\n"
                    "  <li><b>error</b>: 0.0000 (0/5)</li>\n"
                    "  <li><b>number of samples</b>: 5</li>\n"
                    "  <li><b>target label column</b>: target</li>\n"
                    "  <li><b>predicted label column</b>: pred</li>\n"
                    "</ul>"
                ),
                evaluator=Evaluator(
                    {
                        "accuracy": 1.0,
                        "count": 5,
                        "count_correct": 5,
                        "count_incorrect": 0,
                        "error": 0.0,
                    }
                ),
            )
        )
    )


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
        nan_policy=nan_policy,
    )
    generator = AccuracyOutput(state).get_content_generator()
    assert generator.equal(AccuracyContentGenerator.from_state(state))


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_content_generator_lazy_false(nan_policy: str) -> None:
    assert isinstance(
        AccuracyOutput(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
                nan_policy=nan_policy,
            ),
        ).get_content_generator(lazy=False),
        ContentGenerator,
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_evaluator_lazy_true(nan_policy: str) -> None:
    state = AccuracyState(
        y_true=np.array([1, 0, 0, 1, 1]),
        y_pred=np.array([1, 0, 0, 1, 1]),
        y_true_name="target",
        y_pred_name="pred",
        nan_policy=nan_policy,
    )
    evaluator = AccuracyOutput(state).get_evaluator()
    assert evaluator.equal(AccuracyEvaluator(state))


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_output_get_evaluator_lazy_false(nan_policy: str) -> None:
    evaluator = AccuracyOutput(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
            nan_policy=nan_policy,
        ),
    ).get_evaluator(lazy=False)
    assert evaluator.equal(
        Evaluator(
            {"accuracy": 1.0, "count": 5, "count_correct": 5, "count_incorrect": 0, "error": 0.0}
        )
    )
