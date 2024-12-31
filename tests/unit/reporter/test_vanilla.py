from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from arkas.hcg import AccuracyContentGenerator, BaseContentGenerator
from arkas.reporter import Reporter
from arkas.state import AccuracyState

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def generator() -> BaseContentGenerator:
    return AccuracyContentGenerator(
        state=AccuracyState(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([1, 0, 1, 0, 1]),
            y_true_name="target",
            y_pred_name="pred",
        )
    )


##############################
#     Tests for Reporter     #
##############################


def test_reporter_repr(
    tmp_path: Path,
) -> None:
    assert repr(
        Reporter(report_path=tmp_path.joinpath("report.html")),
    ).startswith("Reporter(")


def test_reporter_str(tmp_path: Path) -> None:
    assert str(
        Reporter(report_path=tmp_path.joinpath("report.html")),
    ).startswith("Reporter(")


def test_reporter_generate(tmp_path: Path, generator: BaseContentGenerator) -> None:
    path = tmp_path.joinpath("report.html")
    assert not path.is_file()
    Reporter(report_path=path).generate(generator)
    assert path.is_file()
