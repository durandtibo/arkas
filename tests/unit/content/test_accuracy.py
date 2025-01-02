from __future__ import annotations

import numpy as np

from arkas.content import AccuracyContentGenerator, ContentGenerator
from arkas.content.accuracy import create_template
from arkas.state import AccuracyState

##############################################
#     Tests for AccuracyContentGenerator     #
##############################################


def test_accuracy_content_generator_repr() -> None:
    assert repr(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).startswith("AccuracyContentGenerator(")


def test_accuracy_content_generator_str() -> None:
    assert str(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    ).startswith("AccuracyContentGenerator(")


def test_accuracy_content_generator_equal_true() -> None:
    assert AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_accuracy_content_generator_equal_false_different_state() -> None:
    assert not AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([]),
                y_pred=np.array([]),
                y_true_name="target",
                y_pred_name="pred",
            )
        )
    )


def test_accuracy_content_generator_equal_false_different_nan_policy() -> None:
    assert not AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(
        AccuracyContentGenerator(
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
    assert not AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(42)


def test_accuracy_content_generate_content() -> None:
    assert AccuracyContentGenerator(
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
        "  <li>accuracy: 1.0000 (5/5)</li>\n"
        "  <li>error: 0.0000 (0/5)</li>\n"
        "  <li>number of samples: 5</li>\n"
        "</ul>"
    )


def test_accuracy_content_generate_content_empty() -> None:
    assert AccuracyContentGenerator(
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
        "  <li>accuracy: nan (nan/0)</li>\n"
        "  <li>error: nan (nan/0)</li>\n"
        "  <li>number of samples: 0</li>\n"
        "</ul>"
    )


def test_accuracy_content_generator_generate_body() -> None:
    assert isinstance(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_body(),
        str,
    )


def test_accuracy_content_generator_generate_body_args() -> None:
    assert isinstance(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_accuracy_content_generator_generate_toc() -> None:
    assert isinstance(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_toc(),
        str,
    )


def test_accuracy_content_generator_generate_toc_args() -> None:
    assert isinstance(
        AccuracyContentGenerator(
            state=AccuracyState(
                y_true=np.array([1, 0, 0, 1, 1]),
                y_pred=np.array([1, 0, 1, 0, 1]),
                y_true_name="target",
                y_pred_name="pred",
            )
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_accuracy_content_precompute() -> None:
    assert (
        AccuracyContentGenerator(
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
                "  <li>accuracy: 1.0000 (5/5)</li>\n"
                "  <li>error: 0.0000 (0/5)</li>\n"
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
