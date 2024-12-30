from __future__ import annotations

import numpy as np
from jinja2 import Template

from arkas.hcg import AccuracyContentGenerator
from arkas.hcg.accuracy import create_template
from arkas.state import AccuracyState

##############################################
#     Tests for AccuracyContentGenerator     #
##############################################


def test_accuracy_section_repr() -> None:
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


def test_accuracy_section_str() -> None:
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


def test_accuracy_section_equal_true() -> None:
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


def test_accuracy_section_equal_false_different_state() -> None:
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


def test_accuracy_section_equal_false_different_nan_policy() -> None:
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


def test_accuracy_section_equal_false_different_type() -> None:
    assert not AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    ).equal(42)


def test_accuracy_section_generate_body() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(section.generate_body()).render(), str)


def test_accuracy_section_generate_body_args() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(
        Template(section.generate_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_accuracy_section_generate_body_count_0() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(section.generate_body()).render(), str)


def test_accuracy_section_generate_body_empty() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([]),
            y_pred=np.array([]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(section.generate_body()).render(), str)


def test_accuracy_section_generate_toc() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(Template(section.generate_toc()).render(), str)


def test_accuracy_section_generate_toc_args() -> None:
    section = AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )
    assert isinstance(
        Template(section.generate_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
