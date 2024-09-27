# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from iden.io import PickleSaver

from arkas.evaluator import (
    AccuracyEvaluator,
    BalancedAccuracyEvaluator,
    BinaryAveragePrecisionEvaluator,
    BinaryConfusionMatrixEvaluator,
    BinaryFbetaScoreEvaluator,
    BinaryJaccardEvaluator,
    BinaryPrecisionEvaluator,
    BinaryRecallEvaluator,
    SequentialEvaluator,
)
from arkas.runner import EvaluationRunner
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

    path_metrics = Path.cwd().joinpath("tmp/metrics.pkl")
    runner = EvaluationRunner(
        ingestor=ingestor,
        evaluator=SequentialEvaluator(
            [
                AccuracyEvaluator(y_true="target", y_pred="pred"),
                BalancedAccuracyEvaluator(y_true="target", y_pred="pred"),
                BinaryAveragePrecisionEvaluator(y_true="target", y_score="score"),
                BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred"),
                BinaryFbetaScoreEvaluator(y_true="target", y_pred="pred", betas=[0.5, 1, 2]),
                BinaryJaccardEvaluator(y_true="target", y_pred="pred"),
                BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                BinaryRecallEvaluator(y_true="target", y_pred="pred"),
            ]
        ),
        saver=PickleSaver(),
        path=path_metrics,
    )
    logger.info(f"runner:\n{runner}")
    runner.run()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
