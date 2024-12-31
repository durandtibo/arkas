from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from objectory import OBJECT_TARGET

from arkas.exporter import MetricExporter, SequentialExporter
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


def test_metric_exporter_repr(tmp_path: Path) -> None:
    assert repr(
        SequentialExporter(
            [
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
                {
                    OBJECT_TARGET: "arkas.exporter.FigureExporter",
                    "path": tmp_path.joinpath("figures.pkl"),
                },
            ]
        )
    ).startswith("SequentialExporter(")


def test_metric_exporter_str(tmp_path: Path) -> None:
    assert str(
        SequentialExporter(
            [
                MetricExporter(tmp_path.joinpath("metrics.pkl")),
                {
                    OBJECT_TARGET: "arkas.exporter.FigureExporter",
                    "path": tmp_path.joinpath("figures.pkl"),
                },
            ]
        )
    ).startswith("SequentialExporter(")


def test_metric_exporter_export(tmp_path: Path, output: BaseOutput) -> None:
    SequentialExporter(
        [
            MetricExporter(tmp_path.joinpath("metrics.pkl")),
            {
                OBJECT_TARGET: "arkas.exporter.FigureExporter",
                "path": tmp_path.joinpath("figures.pkl"),
            },
        ]
    ).export(output)
    assert tmp_path.joinpath("metrics.pkl").is_file()
    assert tmp_path.joinpath("figures.pkl").is_file()


def test_metric_exporter_export_empty(output: BaseOutput) -> None:
    SequentialExporter([]).export(output)
