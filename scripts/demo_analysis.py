# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer

from arkas.analyzer import AccuracyAnalyzer
from arkas.exporter import (
    FigureExporter,
    MetricExporter,
    ReportExporter,
    SequentialExporter,
)
from arkas.runner import AnalysisRunner
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    r"""Define the main function."""
    n_samples = 1000
    rng = np.random.default_rng(42)
    ingestor = Ingestor(
        pl.DataFrame(
            {
                "pred": rng.integers(0, 2, (n_samples,)),
                "score": rng.normal(0, 1, (n_samples,)),
                "target": rng.integers(0, 2, (n_samples,)),
            }
        )
    )

    path = Path.cwd().joinpath("tmp")
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred"),
        exporter=SequentialExporter(
            [
                FigureExporter(path=path.joinpath("figures.pkl"), exist_ok=True),
                ReportExporter(path=path.joinpath("report.html"), exist_ok=True),
                MetricExporter(path=path.joinpath("metrics.pkl"), exist_ok=True),
            ]
        ),
    )
    logger.info(f"runner:\n{runner}")
    runner.run()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
