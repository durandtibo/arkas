from __future__ import annotations

from arkas.figure import HtmlFigure

################################
#     Tests for HtmlFigure     #
################################


def test_html_figure_repr() -> None:
    assert repr(HtmlFigure()) == "HtmlFigure()"


def test_html_figure_str() -> None:
    assert str(HtmlFigure()) == "HtmlFigure()"


def test_html_figure_close() -> None:
    HtmlFigure().close()


def test_html_figure_close_multiple() -> None:
    fig = HtmlFigure()
    fig.close()
    fig.close()


def test_html_figure_equal_true() -> None:
    assert HtmlFigure().equal(HtmlFigure())


def test_html_figure_equal_false_different_figure() -> None:
    assert not HtmlFigure().equal(HtmlFigure("meow"))


def test_html_figure_equal_false_different_type() -> None:
    assert not HtmlFigure().equal(42)


def test_html_figure_to_html() -> None:
    assert HtmlFigure("meow").to_html() == "meow"
