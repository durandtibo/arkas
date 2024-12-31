from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer
from iden.io import PickleSaver
from objectory import OBJECT_TARGET

from arkas.evaluator import AccuracyEvaluator
from arkas.runner import EvaluationRunner, is_runner_config, setup_runner

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

######################################
#     Tests for is_runner_config     #
######################################


def test_is_runner_config_true() -> None:
    assert is_runner_config({OBJECT_TARGET: "arkas.runner.EvaluationRunner"})


def test_is_runner_config_false() -> None:
    assert not is_runner_config({OBJECT_TARGET: "collections.Counter"})


##################################
#     Tests for setup_runner     #
##################################


def test_setup_runner_object(tmp_path: Path) -> None:
    runner = EvaluationRunner(
        ingestor=Ingestor(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [3, 2, 0, 1, 0]})),
        transformer=SequentialTransformer(transformers=[]),
        evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        saver=PickleSaver(),
        path=tmp_path.joinpath("metrics.pkl"),
    )
    assert setup_runner(runner) is runner


def test_setup_runner_dict(tmp_path: Path) -> None:
    assert isinstance(
        setup_runner(
            {
                OBJECT_TARGET: "arkas.runner.EvaluationRunner",
                "ingestor": {
                    OBJECT_TARGET: "grizz.ingestor.Ingestor",
                    "frame": pl.DataFrame(
                        {
                            "pred": np.array([3, 2, 0, 1, 0]),
                            "target": np.array([3, 2, 0, 1, 0]),
                        }
                    ),
                },
                "transformer": {OBJECT_TARGET: "grizz.transformer.DropDuplicate"},
                "evaluator": {
                    OBJECT_TARGET: "arkas.evaluator.AccuracyEvaluator",
                    "y_true": "target",
                    "y_pred": "pred",
                },
                "saver": {OBJECT_TARGET: "iden.io.PickleSaver"},
                "path": tmp_path.joinpath("metrics.pkl"),
            }
        ),
        EvaluationRunner,
    )


def test_setup_runner_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_runner({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
