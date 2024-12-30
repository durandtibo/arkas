r"""Contain HTML Content Generators (HCGs)."""

from __future__ import annotations

__all__ = ["AccuracyContentGenerator", "BaseContentGenerator", "ContentGenerator"]

from arkas.hcg.accuracy import AccuracyContentGenerator
from arkas.hcg.base import BaseContentGenerator
from arkas.hcg.vanilla import ContentGenerator
