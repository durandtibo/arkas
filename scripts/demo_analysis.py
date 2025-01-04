# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer

from arkas.analyzer import (
    AccuracyAnalyzer,
    BalancedAccuracyAnalyzer,
    ColumnCooccurrenceAnalyzer,
    DataFrameSummaryAnalyzer,
    MappingAnalyzer,
    PlotColumnAnalyzer,
)
from arkas.exporter import (
    FigureExporter,
    MetricExporter,
    ReportExporter,
    SequentialExporter,
)
from arkas.figure import MatplotlibFigureConfig
from arkas.runner import AnalysisRunner
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    r"""Define the main function."""
    n_samples = 10000
    rng = np.random.default_rng(42)
    ncols = 5
    ingestor = Ingestor(
        pl.DataFrame(
            {
                "pred": rng.integers(0, 2, n_samples),
                "score": rng.normal(0, 1, n_samples),
                "target": rng.integers(0, 2, n_samples),
            }
            | {f"col{i}": rng.integers(0, 2, n_samples) for i in range(ncols)}
        )
    )

    path = Path.cwd().joinpath("tmp")
    figure_config = MatplotlibFigureConfig(dpi=500, figsize=(13, 10))
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=MappingAnalyzer(
            {
                "summary": DataFrameSummaryAnalyzer(),
                "group one": AccuracyAnalyzer(y_true="target", y_pred="pred"),
                "group two": BalancedAccuracyAnalyzer(y_true="target", y_pred="pred"),
                "co-occurrence": ColumnCooccurrenceAnalyzer(
                    columns=[f"col{i}" for i in range(ncols)],
                    ignore_self=True,
                    figure_config=figure_config,
                ),
                # "co-occurrence (symlog)": ColumnCooccurrenceAnalyzer(
                #     columns=[f"col{i}" for i in range(ncols)],
                #     ignore_self=True,
                #     figure_config=MatplotlibFigureConfig(
                #         color_norm=SymLogNorm(linthresh=1), dpi=500, figsize=(13, 10)
                #     ),
                # ),
                "plot columns": PlotColumnAnalyzer(
                    columns=[f"col{i}" for i in range(ncols)], figure_config=figure_config
                ),
            }
        ),
        exporter=SequentialExporter(
            [
                FigureExporter(path=path.joinpath("figures.pkl"), exist_ok=True),
                ReportExporter(path=path.joinpath("report.html"), exist_ok=True),
                MetricExporter(path=path.joinpath("metrics.pkl"), exist_ok=True),
            ]
        ),
        lazy=False,
    )
    logger.info(f"runner:\n{runner}")
    runner.run()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
