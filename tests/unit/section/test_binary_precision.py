from __future__ import annotations

import numpy as np
from jinja2 import Template

from arkas.result import BinaryPrecisionResult, EmptyResult
from arkas.section import BinaryPrecisionSection
from arkas.section.binary_precision import create_section_template

############################################
#     Tests for BinaryPrecisionSection     #
############################################


def test_binary_precision_section_repr() -> None:
    assert repr(
        BinaryPrecisionSection(
            result=BinaryPrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("BinaryPrecisionSection(")


def test_binary_precision_section_str() -> None:
    assert str(
        BinaryPrecisionSection(
            result=BinaryPrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("BinaryPrecisionSection(")


def test_binary_precision_section_equal_true() -> None:
    assert BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    ).equal(
        BinaryPrecisionSection(
            result=BinaryPrecisionResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    )


def test_binary_precision_section_equal_false_different_binary_precision() -> None:
    assert not BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    ).equal(
        BinaryPrecisionSection(
            result=BinaryPrecisionResult(y_true=np.array([]), y_pred=np.array([]))
        )
    )


def test_binary_precision_section_equal_false_different_type() -> None:
    assert not BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    ).equal(42)


def test_binary_precision_section_generate_html_body() -> None:
    section = BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_binary_precision_section_generate_html_body_args() -> None:
    section = BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )
    assert isinstance(
        Template(section.generate_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_binary_precision_section_generate_html_body_count_0() -> None:
    section = BinaryPrecisionSection(
        result=BinaryPrecisionResult(y_true=np.array([]), y_pred=np.array([]))
    )
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_binary_precision_section_generate_html_body_empty() -> None:
    section = BinaryPrecisionSection(result=EmptyResult())
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_binary_precision_section_generate_html_toc() -> None:
    section = BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )
    assert isinstance(Template(section.generate_html_toc()).render(), str)


def test_binary_precision_section_generate_html_toc_args() -> None:
    section = BinaryPrecisionSection(
        result=BinaryPrecisionResult(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
        )
    )
    assert isinstance(
        Template(section.generate_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)
