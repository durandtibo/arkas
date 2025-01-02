from __future__ import annotations

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
    assert ContentGenerator("meow").generate_body() == (
        '<h1 id="">  </h1>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_content_generator_generate_body_args() -> None:
    assert ContentGenerator("meow").generate_body(number="1.", tags=["meow"], depth=1) == (
        '<h2 id="meow">1. meow </h2>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_content_generator_generate_body_depth_1() -> None:
    assert ContentGenerator("meow").generate_body(depth=1) == (
        '<h2 id="">  </h2>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_content_generator_generate_body_depth_2() -> None:
    assert ContentGenerator("meow").generate_body(depth=2) == (
        '<h3 id="">  </h3>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "meow\n"
        '<p style="margin-top: 1rem;">'
    )


def test_content_generator_generate_body_empty() -> None:
    assert ContentGenerator().generate_body() == (
        '<h1 id="">  </h1>\n'
        '<a href="#">Go to top</a>\n'
        '<p style="margin-top: 1rem;">\n'
        "\n"
        '<p style="margin-top: 1rem;">'
    )


def test_content_generator_generate_toc() -> None:
    assert ContentGenerator("meow").generate_toc() == '<li><a href="#"> </a></li>'


def test_content_generator_generate_toc_args() -> None:
    assert (
        ContentGenerator("meow").generate_toc(number="1.", tags=["meow"], depth=1, max_depth=6)
        == '<li><a href="#meow">1. meow</a></li>'
    )


def test_content_generator_generate_toc_too_deep() -> None:
    assert ContentGenerator("meow").generate_toc(number="1.", tags=["meow"], depth=1) == ""


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)
