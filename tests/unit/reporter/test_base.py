from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
from grizz.ingestor import Ingestor
from grizz.transformer import SequentialTransformer
from objectory import OBJECT_TARGET

from arkas.evaluator import AccuracyEvaluator
from arkas.reporter import EvalReporter, is_reporter_config, setup_reporter

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

########################################
#     Tests for is_reporter_config     #
########################################


def test_is_reporter_config_true() -> None:
    assert is_reporter_config(
        {
            OBJECT_TARGET: "arkas.reporter.EvalReporter",
            "ingestor": {
                OBJECT_TARGET: "grizz.ingestor.CsvIngestor",
                "path": "/path/to/data.csv",
            },
            "transformer": {OBJECT_TARGET: "grizz.transformer.DropDuplicate"},
            "evaluator": {
                OBJECT_TARGET: "arkas.evaluator.AccuracyEvaluator",
                "y_true": "target",
                "y_pred": "pred",
            },
            "report_path": "/path/to/report.html",
        }
    )


def test_is_reporter_config_false() -> None:
    assert not is_reporter_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_reporter     #
####################################


def test_setup_reporter_object(tmp_path: Path) -> None:
    reporter = EvalReporter(
        ingestor=Ingestor(pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})),
        transformer=SequentialTransformer(transformers=[]),
        evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        report_path=tmp_path.joinpath("report.html"),
    )
    assert setup_reporter(reporter) is reporter


def test_setup_reporter_dict() -> None:
    assert isinstance(
        setup_reporter(
            {
                OBJECT_TARGET: "arkas.reporter.EvalReporter",
                "ingestor": {
                    OBJECT_TARGET: "grizz.ingestor.CsvIngestor",
                    "path": "/path/to/data.csv",
                },
                "transformer": {OBJECT_TARGET: "grizz.transformer.DropDuplicate"},
                "evaluator": {
                    OBJECT_TARGET: "arkas.evaluator.AccuracyEvaluator",
                    "y_true": "target",
                    "y_pred": "pred",
                },
                "report_path": "/path/to/report.html",
            }
        ),
        EvalReporter,
    )


def test_setup_reporter_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_reporter({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
