from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from coola import objects_are_equal
from iden.io import load_pickle, save_pickle

from arkas.exporter import MetricExporter
from arkas.output import AccuracyOutput, BaseOutput
from arkas.state import AccuracyState

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def output() -> BaseOutput:
    return AccuracyOutput(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 0, 1, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


####################################
#     Tests for MetricExporter     #
####################################


def test_metric_exporter_repr(tmp_path: Path) -> None:
    assert repr(MetricExporter(tmp_path.joinpath("metrics.pkl"))).startswith("MetricExporter(")


def test_metric_exporter_str(tmp_path: Path) -> None:
    assert str(MetricExporter(tmp_path.joinpath("metrics.pkl"))).startswith("MetricExporter(")


def test_metric_exporter_export(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    MetricExporter(path).export(output)
    assert path.is_file()
    assert objects_are_equal(
        load_pickle(path),
        {"accuracy": 1.0, "count_correct": 5, "count_incorrect": 0, "count": 5, "error": 0.0},
    )


def test_metric_exporter_export_show_metrics(
    tmp_path: Path, output: BaseOutput, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    exporter = MetricExporter(path, show_metrics=True)
    with caplog.at_level(level=logging.INFO):
        exporter.export(output)
        assert caplog.messages[-1].startswith("metrics:")
    assert path.is_file()
    assert objects_are_equal(
        load_pickle(path),
        {"accuracy": 1.0, "count_correct": 5, "count_incorrect": 0, "count": 5, "error": 0.0},
    )


def test_metric_exporter_export_exist_ok_false(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    save_pickle({}, path)
    exporter = MetricExporter(path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        exporter.export(output)


def test_metric_exporter_export_exist_ok_true(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("metrics.pkl")
    save_pickle({}, path)
    MetricExporter(path, exist_ok=True).export(output)
    assert path.is_file()
    assert objects_are_equal(
        load_pickle(path),
        {"accuracy": 1.0, "count_correct": 5, "count_incorrect": 0, "count": 5, "error": 0.0},
    )
