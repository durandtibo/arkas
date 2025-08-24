from __future__ import annotations

import numpy as np

from arkas.content import BalancedAccuracyContentGenerator, ContentGenerator
from arkas.content.accuracy import create_template
from arkas.evaluator2 import BalancedAccuracyEvaluator
from arkas.state import AccuracyState

######################################################
#     Tests for BalancedAccuracyContentGenerator     #
######################################################


def test_balanced_accuracy_content_generator_repr() -> None:
    assert repr(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    ).startswith("BalancedAccuracyContentGenerator(")


def test_balanced_accuracy_content_generator_str() -> None:
    assert str(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    ).startswith("BalancedAccuracyContentGenerator(")


def test_balanced_accuracy_content_generator_compute() -> None:
    assert (
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 0, 1, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
        .compute()
        .equal(
            ContentGenerator(
                "<ul>\n"
                "  <li><b>balanced accuracy</b>: 1.0000</li>\n"
                "  <li><b>number of samples</b>: 5</li>\n"
                "  <li><b>target label column</b>: target</li>\n"
                "  <li><b>predicted label column</b>: pred</li>\n"
                "</ul>"
            )
        )
    )


def test_balanced_accuracy_content_generator_equal_true() -> None:
    assert BalancedAccuracyContentGenerator(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    )


def test_balanced_accuracy_content_generator_equal_false_different_state() -> None:
    assert not BalancedAccuracyContentGenerator(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([]),
                    y_pred=np.array([]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    )


def test_balanced_accuracy_content_generator_equal_false_different_type() -> None:
    assert not BalancedAccuracyContentGenerator(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).equal(42)


def test_balanced_accuracy_content_generator_generate_content() -> None:
    assert BalancedAccuracyContentGenerator(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 0, 1, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).generate_content() == (
        "<ul>\n"
        "  <li><b>balanced accuracy</b>: 1.0000</li>\n"
        "  <li><b>number of samples</b>: 5</li>\n"
        "  <li><b>target label column</b>: target</li>\n"
        "  <li><b>predicted label column</b>: pred</li>\n"
        "</ul>"
    )


def test_balanced_accuracy_content_generator_generate_content_empty() -> None:
    assert BalancedAccuracyContentGenerator(
        BalancedAccuracyEvaluator(
            AccuracyState(
                y_true=np.array([]),
                y_pred=np.array([]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).generate_content() == (
        "<ul>\n"
        "  <li><b>balanced accuracy</b>: nan</li>\n"
        "  <li><b>number of samples</b>: 0</li>\n"
        "  <li><b>target label column</b>: target</li>\n"
        "  <li><b>predicted label column</b>: pred</li>\n"
        "</ul>"
    )


def test_balanced_accuracy_content_generator_generate_body() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        ).generate_body(),
        str,
    )


def test_balanced_accuracy_content_generator_generate_body_args() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_balanced_accuracy_content_generator_generate_toc() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        ).generate_toc(),
        str,
    )


def test_balanced_accuracy_content_generator_generate_toc_args() -> None:
    assert isinstance(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_balanced_accuracy_content_generator_from_state() -> None:
    assert BalancedAccuracyContentGenerator.from_state(
        AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        BalancedAccuracyContentGenerator(
            BalancedAccuracyEvaluator(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 1, 0, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
