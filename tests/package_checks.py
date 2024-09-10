from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import polars as pl
from grizz.ingestor import Ingestor
from iden.io import PickleSaver

from arkas.evaluator import AccuracyEvaluator
from arkas.runner import EvaluationRunner
from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def check_evaluator() -> None:
    logger.info("Checking arkas.evaluator package")

    ingestor = Ingestor(
        pl.DataFrame(
            {
                "pred": [1, 0, 0, 1, 1],
                "score": [2, -1, 0, 3, 1],
                "target": [1, 0, 0, 1, 1],
            }
        )
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path_metrics = Path(tmpdir).joinpath("data/metrics.pkl")
        runner = EvaluationRunner(
            ingestor=ingestor,
            evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
            saver=PickleSaver(),
            path=path_metrics,
        )
        logger.info(f"runner:\n{runner}")
        runner.run()
        assert path_metrics.is_file()


def main() -> None:
    check_evaluator()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
