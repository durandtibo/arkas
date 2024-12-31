r"""Contain HTML Content Generators (HCGs)."""

from __future__ import annotations

__all__ = [
    "AccuracyContentGenerator",
    "BaseContentGenerator",
    "ContentGenerator",
    "ContentGeneratorDict",
]

from arkas.hcg.accuracy import AccuracyContentGenerator
from arkas.hcg.base import BaseContentGenerator
from arkas.hcg.mapping import ContentGeneratorDict
from arkas.hcg.vanilla import ContentGenerator
