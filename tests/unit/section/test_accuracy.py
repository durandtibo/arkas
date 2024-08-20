from __future__ import annotations

import numpy as np
from jinja2 import Template

from arkas.result import AccuracyResult, EmptyResult
from arkas.section import AccuracySection
from arkas.section.accuracy import create_section_template

#####################################
#     Tests for AccuracySection     #
#####################################


def test_accuracy_section_repr() -> None:
    assert repr(
        AccuracySection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("AccuracySection(")


def test_accuracy_section_str() -> None:
    assert str(
        AccuracySection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("AccuracySection(")


def test_accuracy_section_equal_true() -> None:
    assert AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(
        AccuracySection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    )


def test_accuracy_section_equal_false_different_accuracy() -> None:
    assert not AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(AccuracySection(result=AccuracyResult(y_true=np.array([]), y_pred=np.array([]))))


def test_accuracy_section_equal_false_different_type() -> None:
    assert not AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(42)


def test_accuracy_section_generate_html_body() -> None:
    section = AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_accuracy_section_generate_html_body_args() -> None:
    section = AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(
        Template(section.generate_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_accuracy_section_generate_html_body_count_0() -> None:
    section = AccuracySection(result=AccuracyResult(y_true=np.array([]), y_pred=np.array([])))
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_accuracy_section_generate_html_body_empty() -> None:
    section = AccuracySection(result=EmptyResult())
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_accuracy_section_generate_html_toc() -> None:
    section = AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(Template(section.generate_html_toc()).render(), str)


def test_accuracy_section_generate_html_toc_args() -> None:
    section = AccuracySection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(
        Template(section.generate_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)
