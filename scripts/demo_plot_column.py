# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer
from grizz.utils.datetime import find_end_datetime

from arkas import analyzer as aa
from arkas.exporter import MetricExporter, ReportExporter, SequentialExporter
from arkas.figure import MatplotlibFigureConfig
from arkas.runner import AnalysisRunner
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_dataframe() -> pl.DataFrame:
    r"""Return a DataFrame."""
    n_samples = 200
    rng = np.random.default_rng(42)
    frame = pl.DataFrame(
        {
            "normal": rng.normal(0, 1, n_samples),
            "uniform": rng.uniform(0, 1, n_samples),
            "cauchy": rng.standard_cauchy(n_samples),
        }
    )
    return frame.with_columns(
        pl.datetime_range(
            start=datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
            end=find_end_datetime(
                datetime(year=2018, month=1, day=1, tzinfo=timezone.utc),
                periods=n_samples - 1,
                interval="1h",
            ),
            interval="1h",
        ).alias("datetime"),
    )


def main() -> None:
    r"""Define the main function."""
    ingestor = Ingestor(get_dataframe())

    path = Path.cwd().joinpath("tmp")
    figure_config = MatplotlibFigureConfig(init={"dpi": 500, "figsize": (14, 8)}, yscale="symlog")
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=aa.MappingAnalyzer(
            {
                "summary": aa.SummaryAnalyzer(),
                "plot columns": aa.PlotColumnAnalyzer(
                    columns=["cauchy", "normal", "uniform"], figure_config=figure_config
                ),
                "temporal plot columns": aa.TemporalPlotColumnAnalyzer(
                    temporal_column="datetime",
                    columns=["cauchy", "normal", "uniform"],
                    figure_config=figure_config,
                ),
                "temporal plot columns (6h)": aa.TemporalPlotColumnAnalyzer(
                    temporal_column="datetime",
                    period="6h",
                    columns=["cauchy", "normal", "uniform"],
                    figure_config=figure_config,
                ),
                "scatter columns": aa.ScatterColumnAnalyzer(
                    x="normal", y="cauchy", color="uniform", figure_config=figure_config
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
