r"""Contain HTML content generators."""

from __future__ import annotations

__all__ = [
    "AccuracyContentGenerator",
    "BalancedAccuracyContentGenerator",
    "BaseContentGenerator",
    "ColumnCooccurrenceContentGenerator",
    "ContentGenerator",
    "ContentGeneratorDict",
    "DataFrameSummaryContentGenerator",
    "PlotColumnContentGenerator",
    "ScatterColumnContentGenerator",
    "TemporalPlotColumnContentGenerator",
]

from arkas.content.accuracy import AccuracyContentGenerator
from arkas.content.balanced_accuracy import BalancedAccuracyContentGenerator
from arkas.content.base import BaseContentGenerator
from arkas.content.column_cooccurrence import ColumnCooccurrenceContentGenerator
from arkas.content.frame_summary import DataFrameSummaryContentGenerator
from arkas.content.mapping import ContentGeneratorDict
from arkas.content.plot_column import PlotColumnContentGenerator
from arkas.content.scatter_column import ScatterColumnContentGenerator
from arkas.content.temporal_plot_column import TemporalPlotColumnContentGenerator
from arkas.content.vanilla import ContentGenerator
