# noqa: INP001
r"""Contain a demo example."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from coola.nested import to_flat_dict
from coola.utils import str_mapping
from iden.io import PickleSaver, load_pickle

from arkas.data.ingestor import Ingestor
from arkas.evaluator import (
    AccuracyEvaluator,
    BalancedAccuracyEvaluator,
    BinaryAveragePrecisionEvaluator,
    BinaryConfusionMatrixEvaluator,
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
        {
            "pred": rng.integers(0, 2, (n_samples,)),
            "score": rng.normal(0, 1, (n_samples,)),
            "target": rng.integers(0, 2, (n_samples,)),
        }
    )

    path_metrics = Path.cwd().joinpath("tmp/metrics.pkl")
    runner = EvaluationRunner(
        ingestor=ingestor,
        evaluator=SequentialEvaluator(
            [
                AccuracyEvaluator(y_true="target", y_pred="pred"),
                BalancedAccuracyEvaluator(y_true="target", y_pred="pred"),
                BinaryPrecisionEvaluator(y_true="target", y_pred="pred"),
                BinaryRecallEvaluator(y_true="target", y_pred="pred"),
                BinaryJaccardEvaluator(y_true="target", y_pred="pred"),
                BinaryConfusionMatrixEvaluator(y_true="target", y_pred="pred"),
                BinaryAveragePrecisionEvaluator(y_true="target", y_score="score"),
            ]
        ),
        saver=PickleSaver(),
        path=path_metrics,
    )
    logger.info(f"runner:\n{runner}")
    runner.run()

    metrics = load_pickle(path_metrics)
    logger.info(f"metrics:\n{str_mapping(to_flat_dict(metrics), sorted_keys=True)}")


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
