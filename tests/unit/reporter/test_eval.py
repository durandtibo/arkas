from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from grizz.ingestor import BaseIngestor, Ingestor
from grizz.transformer import SequentialTransformer

from arkas.evaluator import AccuracyEvaluator
from arkas.reporter import EvalReporter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ingestor() -> BaseIngestor:
    return Ingestor(pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]}))


##################################
#     Tests for EvalReporter     #
##################################


def test_eval_reporter_repr(tmp_path: Path, ingestor: BaseIngestor) -> None:
    assert repr(
        EvalReporter(
            ingestor=ingestor,
            transformer=SequentialTransformer(transformers=[]),
            evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
            report_path=tmp_path.joinpath("report.html"),
        ),
    ).startswith("EvalReporter(")


def test_eval_reporter_str(tmp_path: Path, ingestor: BaseIngestor) -> None:
    assert str(
        EvalReporter(
            ingestor=ingestor,
            transformer=SequentialTransformer(transformers=[]),
            evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
            report_path=tmp_path.joinpath("report.html"),
        ),
    ).startswith("EvalReporter(")


def test_eval_reporter_generate(tmp_path: Path, ingestor: BaseIngestor) -> None:
    path = tmp_path.joinpath("report.html")
    assert not path.is_file()
    EvalReporter(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        report_path=path,
    ).compute()
    assert path.is_file()


def test_eval_reporter_generate_empty(tmp_path: Path) -> None:
    path = tmp_path.joinpath("report.html")
    assert not path.is_file()
    EvalReporter(
        ingestor=Ingestor(pl.DataFrame()),
        transformer=SequentialTransformer(transformers=[]),
        evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        report_path=path,
    ).compute()
    assert path.is_file()
