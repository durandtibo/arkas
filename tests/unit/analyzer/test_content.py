from __future__ import annotations

import polars as pl

from arkas.analyzer import ContentAnalyzer
from arkas.content import ContentGenerator
from arkas.evaluator2 import Evaluator
from arkas.output import ContentOutput, Output
from arkas.plotter import Plotter

#####################################
#     Tests for ContentAnalyzer     #
#####################################


def test_content_analyzer_repr() -> None:
    assert repr(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_str() -> None:
    assert str(ContentAnalyzer(content="meow")).startswith("ContentAnalyzer(")


def test_content_analyzer_analyze() -> None:
    assert (
        ContentAnalyzer(content="meow")
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
        .equal(ContentOutput(content="meow"))
    )


def test_content_analyzer_analyze_lazy_false() -> None:
    assert (
        ContentAnalyzer(content="meow")
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}), lazy=False)
        .equal(Output(content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()))
    )
