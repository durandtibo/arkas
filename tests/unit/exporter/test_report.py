from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iden.io import save_pickle, save_text

from arkas.exporter import ReportExporter
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
#     Tests for ReportExporter     #
####################################


def test_report_exporter_repr(tmp_path: Path) -> None:
    assert repr(ReportExporter(tmp_path.joinpath("report.html")))


def test_report_exporter_str(tmp_path: Path) -> None:
    assert str(ReportExporter(tmp_path.joinpath("report.html")))


def test_report_exporter_export(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("report.html")
    ReportExporter(path).export(output)
    assert path.is_file()


def test_report_exporter_export_exist_ok_false(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("report.html")
    save_text("", path)
    exporter = ReportExporter(path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        exporter.export(output)


def test_report_exporter_export_exist_ok_true(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("report.html")
    save_pickle({}, path)
    ReportExporter(path, exist_ok=True).export(output)
    assert path.is_file()


@pytest.mark.parametrize("max_toc_depth", [1, 2, 3])
def test_report_exporter_export_max_toc_depth(
    tmp_path: Path, output: BaseOutput, max_toc_depth: int
) -> None:
    path = tmp_path.joinpath("report.html")
    ReportExporter(path, max_toc_depth=max_toc_depth).export(output)
    assert path.is_file()
