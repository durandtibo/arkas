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
from arkas.exporter import (
    FigureExporter,
    MetricExporter,
    ReportExporter,
    SequentialExporter,
)
from arkas.figure import MatplotlibFigureConfig
from arkas.runner import AnalysisRunner
from arkas.utils.array import rand_replace
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def get_dataframe() -> pl.DataFrame:
    r"""Return a DataFrame."""
    n_samples = 20000
    ncols = 10
    rng = np.random.default_rng(42)
    frame = pl.DataFrame(
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
    figure_config = MatplotlibFigureConfig(init={"dpi": 500, "figsize": (14, 6)})
    runner = AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=aa.MappingAnalyzer(
            {
                "summary": aa.SummaryAnalyzer(),
                "numeric summary": aa.NumericSummaryAnalyzer(),
                "correlation": aa.ColumnCorrelationAnalyzer(target_column="col1"),
                "correlation col1 - col2": aa.CorrelationAnalyzer(
                    x="col1", y="col2", figure_config=figure_config
                ),
                "null values": aa.NullValueAnalyzer(figure_config=figure_config),
                "temporal null values": aa.TemporalNullValueAnalyzer(
                    temporal_column="datetime", period="1mo", figure_config=figure_config
                ),
                "continuous": aa.ContinuousColumnAnalyzer(
                    column="col1",
                    figure_config=MatplotlibFigureConfig(
                        init={"dpi": 500, "figsize": (14, 6)},
                        nbins=201,
                        xmin="q0.001",
                        xmax="q0.999",
                    ),
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
