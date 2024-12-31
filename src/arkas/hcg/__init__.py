r"""Contain HTML Content Generators (HCGs)."""

from __future__ import annotations

__all__ = [
    "AccuracyContentGenerator",
    "BalancedAccuracyContentGenerator",
    "BaseContentGenerator",
    "ContentGenerator",
    "ContentGeneratorDict",
]

from arkas.hcg.accuracy import AccuracyContentGenerator
from arkas.hcg.balanced_accuracy import BalancedAccuracyContentGenerator
from arkas.hcg.base import BaseContentGenerator
from arkas.hcg.mapping import ContentGeneratorDict
from arkas.hcg.vanilla import ContentGenerator
