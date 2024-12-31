r"""Contain HTML Content Generators (HCGs)."""

from __future__ import annotations

__all__ = [
    "AccuracyContentGenerator",
    "BalancedAccuracyContentGenerator",
    "BaseContentGenerator",
    "ContentGenerator",
    "ContentGeneratorDict",
]

from arkas.content.accuracy import AccuracyContentGenerator
from arkas.content.balanced_accuracy import BalancedAccuracyContentGenerator
from arkas.content.base import BaseContentGenerator
from arkas.content.mapping import ContentGeneratorDict
from arkas.content.vanilla import ContentGenerator
