from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pytest
from iden.io import load_pickle, save_pickle

from arkas.evaluator2 import Evaluator
from arkas.exporter import FigureExporter
from arkas.hcg import ContentGenerator
from arkas.output import BaseOutput, Output
from arkas.plotter import Plotter

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def output() -> BaseOutput:
    return Output(
        generator=ContentGenerator("meow"),
        evaluator=Evaluator(),
        plotter=Plotter({"fig": plt.subplots()[0]}),
    )


####################################
#     Tests for FigureExporter     #
####################################


def test_figure_exporter_repr(tmp_path: Path) -> None:
    assert repr(FigureExporter(tmp_path.joinpath("figures.pkl")))


def test_figure_exporter_str(tmp_path: Path) -> None:
    assert str(FigureExporter(tmp_path.joinpath("figures.pkl")))


def test_figure_exporter_export(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("figures.pkl")
    FigureExporter(path).export(output)
    assert path.is_file()
    figures = load_pickle(path)
    assert len(figures) == 1
    assert isinstance(figures["fig"], plt.Figure)


def test_figure_exporter_export_exist_ok_false(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("figures.pkl")
    save_pickle({}, path)
    exporter = FigureExporter(path)
    with pytest.raises(FileExistsError, match="path .* already exists."):
        exporter.export(output)


def test_figure_exporter_export_exist_ok_true(tmp_path: Path, output: BaseOutput) -> None:
    path = tmp_path.joinpath("figures.pkl")
    save_pickle({}, path)
    FigureExporter(path, exist_ok=True).export(output)
    assert path.is_file()
    figures = load_pickle(path)
    assert len(figures) == 1
    assert isinstance(figures["fig"], plt.Figure)
