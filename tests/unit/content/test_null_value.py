from __future__ import annotations

import numpy as np

from arkas.content import ContentGenerator, NullValueContentGenerator
from arkas.content.null_value import create_template
from arkas.state import NullValueState

###############################################
#     Tests for NullValueContentGenerator     #
###############################################


def test_null_value_content_generator_repr() -> None:
    assert repr(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValueContentGenerator(")


def test_null_value_content_generator_str() -> None:
    assert str(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("NullValueContentGenerator(")


def test_null_value_content_generator_compute() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).compute(),
        ContentGenerator,
    )


def test_null_value_content_generator_equal_true() -> None:
    assert NullValueContentGenerator(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_null_value_content_generator_equal_false_different_state() -> None:
    assert not NullValueContentGenerator(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        NullValueContentGenerator(
            NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[])
        )
    )


def test_null_value_content_generator_equal_false_different_type() -> None:
    assert not NullValueContentGenerator(
        NullValueState(
            null_count=np.array([1, 2, 3]),
            total_count=np.array([7, 7, 7]),
            columns=["col1", "col2", "col3"],
        )
    ).equal(42)


def test_null_value_content_generator_generate_content() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).generate_content(),
        str,
    )


def test_null_value_content_generator_generate_content_empty() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(null_count=np.array([]), total_count=np.array([]), columns=[])
        ).generate_content(),
        str,
    )


def test_null_value_content_generator_generate_body() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).generate_body(),
        str,
    )


def test_null_value_content_generator_generate_body_args() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_null_value_content_generator_generate_toc() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).generate_toc(),
        str,
    )


def test_null_value_content_generator_generate_toc_args() -> None:
    assert isinstance(
        NullValueContentGenerator(
            NullValueState(
                null_count=np.array([1, 2, 3]),
                total_count=np.array([7, 7, 7]),
                columns=["col1", "col2", "col3"],
            )
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
