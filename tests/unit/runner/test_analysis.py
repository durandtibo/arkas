from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest
from coola import objects_are_equal
from grizz.ingestor import BaseIngestor, Ingestor
from grizz.transformer import SequentialTransformer
from iden.io import load_pickle

from arkas.analyzer import AccuracyAnalyzer
from arkas.exporter import MetricExporter
from arkas.runner import AnalysisRunner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ingestor() -> BaseIngestor:
    return Ingestor(
        pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([3, 2, 0, 1, 0])})
    )


####################################
#     Tests for AnalysisRunner     #
####################################


def test_analysis_runner_repr(tmp_path: Path, ingestor: BaseIngestor) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    assert repr(
        AnalysisRunner(
            ingestor=ingestor,
            transformer=SequentialTransformer(transformers=[]),
            analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred"),
            exporter=MetricExporter(path=path),
        )
    ).startswith("AnalysisRunner(")


def test_analysis_runner_str(tmp_path: Path, ingestor: BaseIngestor) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    assert str(
        AnalysisRunner(
            ingestor=ingestor,
            transformer=SequentialTransformer(transformers=[]),
            analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred"),
            exporter=MetricExporter(path=path),
        )
    ).startswith("AnalysisRunner(")


def test_analysis_runner_run(tmp_path: Path, ingestor: BaseIngestor) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    AnalysisRunner(
        ingestor=ingestor,
        transformer=SequentialTransformer(transformers=[]),
        analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred"),
        exporter=MetricExporter(path=path),
    ).run()
    assert path.is_file()
    assert objects_are_equal(
        load_pickle(path),
        {"accuracy": 1.0, "count_correct": 5, "count_incorrect": 0, "count": 5, "error": 0.0},
    )
