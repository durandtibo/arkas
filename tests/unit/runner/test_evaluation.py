from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer
from iden.io import PickleSaver, load_pickle

from arkas.evaluator import AccuracyEvaluator
from arkas.runner import EvaluationRunner

if TYPE_CHECKING:
    from pathlib import Path

######################################
#     Tests for EvaluationRunner     #
######################################


def test_evaluation_runner_repr(tmp_path: Path) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    assert repr(
        EvaluationRunner(
            ingestor=Ingestor(
                pl.DataFrame(
                    {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}
                )
            ),
            transformer=SequentialTransformer(transformers=[]),
            evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
            saver=PickleSaver(),
            path=path,
        )
    ).startswith("EvaluationRunner(")


def test_evaluation_runner_str(tmp_path: Path) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    assert str(
        EvaluationRunner(
            ingestor=Ingestor(
                pl.DataFrame(
                    {"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])}
                )
            ),
            transformer=SequentialTransformer(transformers=[]),
            evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
            saver=PickleSaver(),
            path=path,
        )
    ).startswith("EvaluationRunner(")


@pytest.mark.parametrize("show_metrics", [True, False])
def test_evaluation_runner_evaluate(tmp_path: Path, show_metrics: bool) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    runner = EvaluationRunner(
        ingestor=Ingestor(
            pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])})
        ),
        transformer=SequentialTransformer(transformers=[]),
        evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        saver=PickleSaver(),
        path=path,
        show_metrics=show_metrics,
    )
    assert not path.is_file()
    runner.run()
    assert path.is_file()
    assert objects_are_equal(
        load_pickle(path),
        {"accuracy": 1.0, "count_correct": 5, "count_incorrect": 0, "count": 5, "error": 0.0},
    )
