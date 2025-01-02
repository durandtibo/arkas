from __future__ import annotations

from jinja2 import Template

from arkas.content import ContentGenerator
from arkas.content.accuracy import create_template

######################################
#     Tests for ContentGenerator     #
######################################


def test_content_generator_repr() -> None:
    assert repr(ContentGenerator("meow")).startswith("ContentGenerator(")


def test_content_generator_str() -> None:
    assert str(ContentGenerator("meow")).startswith("ContentGenerator(")


def test_content_generator_equal_true() -> None:
    assert ContentGenerator("meow").equal(ContentGenerator("meow"))


def test_content_generator_equal_false_different_state() -> None:
    assert not ContentGenerator("meow").equal(ContentGenerator("miaou"))


def test_content_generator_equal_false_different_type() -> None:
    assert not ContentGenerator("meow").equal(42)


def test_content_generator_generate_content() -> None:
    assert ContentGenerator("meow").generate_content() == "meow"


def test_content_generator_generate_body() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_content_generator_generate_body_args() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(
        Template(generator.generate_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_content_generator_generate_body_count_0() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_content_generator_generate_body_empty() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(Template(generator.generate_body()).render(), str)


def test_content_generator_generate_toc() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(Template(generator.generate_toc()).render(), str)


def test_content_generator_generate_toc_args() -> None:
    generator = ContentGenerator("meow")
    assert isinstance(
        Template(generator.generate_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
