from __future__ import annotations

import numpy as np

from arkas.content import BalancedAccuracyContentGenerator, ContentGenerator
from arkas.content.accuracy import create_template
from arkas.state import AccuracyState

######################################################
#     Tests for BalancedAccuracyContentGenerator     #
######################################################


def test_balanced_accuracy_content_generator_repr() -> None:
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


def test_balanced_accuracy_content_generator_str() -> None:
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


def test_balanced_accuracy_content_generator_equal_true() -> None:
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


def test_balanced_accuracy_content_generator_equal_false_different_state() -> None:
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


def test_balanced_accuracy_content_generator_equal_false_different_nan_policy() -> None:
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


def test_balanced_accuracy_content_generator_equal_false_different_type() -> None:
    assert not BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(42)


def test_balanced_accuracy_content_generator_generate_content() -> None:
    assert BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).generate_content() == (
        "<ul>\n"
        "  <li>column with target labels: target</li>\n"
        "  <li>column with predicted labels: pred</li>\n"
        "  <li>balanced accuracy: 1.0000</li>\n"
        "  <li>number of samples: 5</li>\n"
        "</ul>"
    )


def test_balanced_accuracy_content_generator_generate_content_empty() -> None:
    assert BalancedAccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).generate_content() == (
        "<ul>\n"
        "  <li>column with target labels: target</li>\n"
        "  <li>column with predicted labels: pred</li>\n"
        "  <li>balanced accuracy: nan</li>\n"
        "  <li>number of samples: 0</li>\n"
        "</ul>"
    )


def test_balanced_accuracy_content_generator_generate_body() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_body(),
        str,
    )


def test_balanced_accuracy_content_generator_generate_body_args() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_balanced_accuracy_content_generator_generate_toc() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_toc(),
        str,
    )


def test_balanced_accuracy_content_generator_generate_toc_args() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_balanced_accuracy_content_generator_precompute() -> None:
    assert (
        BalancedAccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
        .precompute()
        .equal(
            ContentGenerator(
                "<ul>\n"
                "  <li>column with target labels: target</li>\n"
                "  <li>column with predicted labels: pred</li>\n"
                "  <li>balanced accuracy: 1.0000</li>\n"
                "  <li>number of samples: 5</li>\n"
                "</ul>"
            )
        )
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
