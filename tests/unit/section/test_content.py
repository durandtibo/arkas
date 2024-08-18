from __future__ import annotations

from jinja2 import Template

from arkas.section import ContentSection
from arkas.section.content import create_section_template

####################################
#     Tests for ContentSection     #
####################################


def test_content_section_str() -> None:
    assert str(ContentSection(content="meow")).startswith("ContentSection(")


def test_content_section_generate_html_body() -> None:
    section = ContentSection(content="meow")
    assert isinstance(Template(section.generate_html_body()).render(), str)


def test_content_section_generate_html_body_args() -> None:
    section = ContentSection(content="meow")
    assert isinstance(
        Template(section.generate_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


def test_content_section_generate_html_toc() -> None:
    section = ContentSection(content="meow")
    assert isinstance(Template(section.generate_html_toc()).render(), str)


def test_content_section_generate_html_toc_args() -> None:
    section = ContentSection(content="meow")
    assert isinstance(
        Template(section.generate_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)
