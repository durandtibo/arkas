from __future__ import annotations

import numpy as np
from jinja2 import Template

from arkas.content import BalancedAccuracyContentGenerator
from arkas.content.accuracy import create_template
from arkas.state import AccuracyState

######################################################
#     Tests for BalancedAccuracyContentGenerator     #
######################################################


def test_accuracy_content_generator_repr() -> None:
    assert repr(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).startswith("BalancedAccuracyContentGenerator(")


def test_accuracy_content_generator_str() -> None:
    assert str(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).startswith("BalancedAccuracyContentGenerator(")


def test_accuracy_content_generator_equal_true() -> None:
    assert BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_accuracy_content_generator_equal_false_different_state() -> None:
    assert not BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([]),
                y_pred=np.array([]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_accuracy_content_generator_equal_false_different_nan_policy() -> None:
    assert not BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            ),
            nan_policy="omit",
        )
    )


def test_accuracy_content_generator_equal_false_different_type() -> None:
    assert not BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(42)


def test_accuracy_content_generator_generate_body() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_accuracy_content_generator_generate_body_args() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(
        Template(generator.generate_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_accuracy_content_generator_generate_body_count_0() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_accuracy_content_generator_generate_body_empty() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_accuracy_content_generator_generate_toc() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(generator.generate_toc()).render(), str)


def test_accuracy_content_generator_generate_toc_args() -> None:
    generator = BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(
        Template(generator.generate_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
