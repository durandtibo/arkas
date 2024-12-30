from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from jinja2 import Template
from matplotlib import pyplot as plt

from arkas.result import AccuracyResult, EmptyResult
from arkas.section import ResultSection
from arkas.section.accuracy import create_section_template
from arkas.section.result import (
    create_figures,
    create_table_metrics,
    create_table_metrics_row,
)

###################################
#     Tests for ResultSection     #
###################################


def test_result_section_repr() -> None:
    assert repr(
        ResultSection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("ResultSection(")


def test_result_section_str() -> None:
    assert str(
        ResultSection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    ).startswith("ResultSection(")


def test_result_section_equal_true() -> None:
    assert ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(
        ResultSection(
            result=AccuracyResult(
                y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1])
            )
        )
    )


def test_result_section_equal_false_different_accuracy() -> None:
    assert not ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(ResultSection(result=AccuracyResult(y_true=np.array([]), y_pred=np.array([]))))


def test_result_section_equal_false_different_type() -> None:
    assert not ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    ).equal(42)


def test_result_section_generate_html_body() -> None:
    section = ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_result_section_generate_html_body_args() -> None:
    section = ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(
        Template(section.generate_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_result_section_generate_html_body_count_0() -> None:
    section = ResultSection(result=AccuracyResult(y_true=np.array([]), y_pred=np.array([])))
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_result_section_generate_html_body_empty() -> None:
    section = ResultSection(result=EmptyResult())
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_result_section_generate_html_toc() -> None:
    section = ResultSection(
        result=AccuracyResult(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([1, 0, 1, 0, 1]))
    )
    assert isinstance(Template(section.generate_html_toc()).render(), str)


def test_result_section_generate_html_toc_args() -> None:
    section = ResultSection(
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


##########################################
#     Tests for create_table_metrics     #
##########################################


def test_create_table_metrics() -> None:
    assert isinstance(
        create_table_metrics(
            {"accuracy": 0.42, "f1": 0.57, "subgroup": {"accuracy": 0.42, "f1": 0.57}}
        ),
        str,
    )


def test_create_table_metrics_empty() -> None:
    assert isinstance(create_table_metrics({}), str)


##############################################
#     Tests for create_table_metrics_row     #
##############################################


@pytest.mark.parametrize("value", [1, 1.2, "abc", np.array([1, 2, 3]), [1, 2, 3]])
def test_create_table_metrics_row(value: Any) -> None:
    assert isinstance(create_table_metrics_row(name="meow", value=value), str)


####################################
#     Tests for create_figures     #
####################################


def test_create_figures() -> None:
    fig, _ = plt.subplots()
    assert isinstance(create_figures({"accuracy": fig}), str)


def test_create_figures_empty() -> None:
    assert isinstance(create_figures({}), str)
