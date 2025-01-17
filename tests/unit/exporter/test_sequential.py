from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from objectory import OBJECT_TARGET

from arkas.exporter import MetricExporter, ReportExporter, SequentialExporter
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


########################################
#     Tests for SequentialExporter     #
########################################


def test_sequential_exporter_repr(tmp_path: Path) -> None:
    assert repr(
        SequentialExporter(
            [
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
                {
                    OBJECT_TARGET: "arkas.exporter.ReportExporter",
                    "path": tmp_path.joinpath("report.html"),
                },
            ]
        )
    ).startswith("SequentialExporter(")


def test_sequential_exporter_str(tmp_path: Path) -> None:
    assert str(
        SequentialExporter(
            [
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
                {
                    OBJECT_TARGET: "arkas.exporter.ReportExporter",
                    "path": tmp_path.joinpath("report.html"),
                },
            ]
        )
    ).startswith("SequentialExporter(")


def test_sequential_exporter_equal_true(tmp_path: Path) -> None:
    assert SequentialExporter(
        [
            MetricExporter(tmp_path.joinpath("metrics.pkl")),
            ReportExporter(tmp_path.joinpath("report.html")),
        ]
    ).equal(
        SequentialExporter(
            [
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
                ReportExporter(tmp_path.joinpath("report.html")),
            ]
        )
    )


def test_sequential_exporter_equal_false_different_exporters(tmp_path: Path) -> None:
    assert not SequentialExporter(
        [
            MetricExporter(tmp_path.joinpath("metrics.pkl")),
            ReportExporter(tmp_path.joinpath("report.html")),
        ]
    ).equal(
        SequentialExporter(
            [
                ReportExporter(tmp_path.joinpath("report.html")),
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
            ]
        )
    )


def test_sequential_exporter_equal_false_different_type(tmp_path: Path) -> None:
    assert not SequentialExporter(
        [
            MetricExporter(tmp_path.joinpath("metrics.pkl")),
            ReportExporter(tmp_path.joinpath("report.html")),
        ]
    ).equal(42)


def test_sequential_exporter_export(tmp_path: Path, output: BaseOutput) -> None:
    SequentialExporter(
        [
            MetricExporter(tmp_path.joinpath("metrics.pkl")),
            {
                OBJECT_TARGET: "arkas.exporter.ReportExporter",
                "path": tmp_path.joinpath("report.html"),
            },
        ]
    ).export(output)
    assert tmp_path.joinpath("metrics.pkl").is_file()
    assert tmp_path.joinpath("report.html").is_file()


def test_sequential_exporter_export_empty(output: BaseOutput) -> None:
    SequentialExporter([]).export(output)
