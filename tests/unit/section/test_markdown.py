from __future__ import annotations

from jinja2 import Template

from arkas.section import MarkdownSection
from arkas.section.markdown import create_section_template
from arkas.testing import markdown_available

#####################################
#     Tests for MarkdownSection     #
#####################################


@markdown_available
def test_markdown_section_str() -> None:
    assert str(MarkdownSection(desc="meow")).startswith("MarkdownSection(")


@markdown_available
def test_markdown_section_generate_html_body() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(Template(section.generate_html_body()).render(), str)


@markdown_available
def test_markdown_section_generate_html_body_args() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(
        Template(section.generate_html_body(number="1.", tags=["meow"], depth=1)).render(), str
    )


@markdown_available
def test_markdown_section_generate_html_body_empty() -> None:
    section = MarkdownSection(desc="")
    assert isinstance(Template(section.generate_html_body()).render(), str)


@markdown_available
def test_markdown_section_generate_html_toc() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(Template(section.generate_html_toc()).render(), str)


@markdown_available
def test_markdown_section_generate_html_toc_args() -> None:
    section = MarkdownSection(desc="### Hello Cat!")
    assert isinstance(
        Template(section.generate_html_toc(number="1.", tags=["meow"], depth=1)).render(), str
    )


#############################################
#     Tests for create_section_template     #
#############################################


def test_create_section_template() -> None:
    assert isinstance(create_section_template(), str)
