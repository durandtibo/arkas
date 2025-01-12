# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import FirstRowTransformer, SequentialTransformer

from arkas import analyzer as aa
from arkas.exporter import MetricExporter, ReportExporter, SequentialExporter
from arkas.figure import MatplotlibFigureConfig
from arkas.runner import AnalysisRunner
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_dataframe() -> pl.DataFrame:
    r"""Return a DataFrame."""
    n_samples = 1_000_000
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "normal": rng.normal(0, 1, n_samples),
            "uniform": rng.uniform(0, 1, n_samples),
            "cauchy": rng.standard_cauchy(n_samples),
            "normal2": rng.normal(0, 1, n_samples),
        }
    )


def main() -> None:
    r"""Define the main function."""
    ingestor = Ingestor(get_dataframe())

    path = Path.cwd().joinpath("tmp")
    figure_config = MatplotlibFigureConfig(
        init={"dpi": 300, "figsize": (14, 6)},
        xmin="q0.001",
        xmax="q0.999",
        ymin="q0.001",
        ymax="q0.999",
    )
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=aa.MappingAnalyzer(
            {
                "summary": aa.SummaryAnalyzer(),
                "numeric summary": aa.NumericSummaryAnalyzer(),
                "scatter columns": aa.ScatterColumnAnalyzer(
                    x="normal",
                    y="cauchy",
                    color="uniform",
                    figure_config=figure_config,
                ),
                "scatter 1k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=1_000),
                    analyzer=aa.ScatterColumnAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "scatter 10k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=10_000),
                    analyzer=aa.ScatterColumnAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "scatter 100k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=100_000),
                    analyzer=aa.ScatterColumnAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "scatter 1M": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=1_000_000),
                    analyzer=aa.ScatterColumnAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "correlation 1k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=1_000),
                    analyzer=aa.CorrelationAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "correlation 10k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=10_000),
                    analyzer=aa.CorrelationAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "correlation 100k": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=100_000),
                    analyzer=aa.CorrelationAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
                "correlation 1M": aa.TransformAnalyzer(
                    transformer=FirstRowTransformer(n=1_000_000),
                    analyzer=aa.CorrelationAnalyzer(
                        x="normal",
                        y="normal2",
                        figure_config=figure_config,
                    ),
                ),
            }
        ),
        exporter=SequentialExporter(
            [
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
