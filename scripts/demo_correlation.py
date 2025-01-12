# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer

from arkas import analyzer as aa
from arkas.exporter import MetricExporter, ReportExporter, SequentialExporter
from arkas.figure import MatplotlibFigureConfig
from arkas.runner import AnalysisRunner
from arkas.utils.array import rand_replace
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_dataframe(ncols: int = 100) -> pl.DataFrame:
    r"""Return a DataFrame."""
    n_samples = 100_000
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            f"col{i}": rand_replace(
                rng.normal(size=(n_samples,)),
                value=None,
                prob=i / ncols,
                rng=rng,
            ).tolist()
            for i in range(ncols)
        }
    )


def main() -> None:
    r"""Define the main function."""
    ncols = 100
    ingestor = Ingestor(get_dataframe(ncols))

    path = Path.cwd().joinpath("tmp")
    figure_config = MatplotlibFigureConfig(init={"dpi": 300, "figsize": (14, 6)})
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=aa.MappingAnalyzer(
            {
                "summary": aa.SummaryAnalyzer(),
                "numeric summary": aa.NumericSummaryAnalyzer(),
                "correlation (pearson)": aa.ColumnCorrelationAnalyzer(
                    target_column="col0", sort_metric="pearson_coeff"
                ),
                "correlation (spearman)": aa.ColumnCorrelationAnalyzer(
                    target_column="col0", sort_metric="spearman_coeff"
                ),
            }
            | {
                f"correlation col{i}": aa.CorrelationAnalyzer(
                    x="col0", y=f"col{i}", figure_config=figure_config
                )
                for i in range(1, ncols)
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
