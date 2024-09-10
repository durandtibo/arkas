# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor

from arkas.evaluator import AccuracyEvaluator
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
                "target": rng.integers(0, 2, (n_samples,)),
            }
        )
    )
    logger.info("Ingesting data...")
    data = ingestor.ingest()

    evaluator = AccuracyEvaluator(y_true="target", y_pred="pred")
    logger.info("Evaluating metrics")
    result = evaluator.evaluate(data)
    logger.info(f"metrics: {result.compute_metrics()}")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
