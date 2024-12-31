r"""Contain DataFrame analyzers."""

from __future__ import annotations

__all__ = [
    "AccuracyAnalyzer",
    "BaseAnalyzer",
    "BaseTruePredAnalyzer",
    "is_analyzer_config",
    "setup_analyzer",
]

from arkas.analyzer.accuracy import AccuracyAnalyzer
from arkas.analyzer.base import BaseAnalyzer, is_analyzer_config, setup_analyzer
from arkas.analyzer.columns import BaseTruePredAnalyzer
