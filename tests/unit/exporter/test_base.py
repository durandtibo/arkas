from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from arkas.exporter import MetricExporter, is_exporter_config, setup_exporter

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

########################################
#     Tests for is_exporter_config     #
########################################


def test_is_exporter_config_true() -> None:
    assert is_exporter_config(
        {
            OBJECT_TARGET: "arkas.exporter.MetricExporter",
            "path": "/path/to/data.csv",
        }
    )


def test_is_exporter_config_false() -> None:
    assert not is_exporter_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_exporter     #
####################################


def test_setup_exporter_object(tmp_path: Path) -> None:
    exporter = MetricExporter(
        path=tmp_path.joinpath("report.html"),
    )
    assert setup_exporter(exporter) is exporter


def test_setup_exporter_dict() -> None:
    assert isinstance(
        setup_exporter(
            {
                OBJECT_TARGET: "arkas.exporter.MetricExporter",
                "path": "/path/to/data.csv",
            }
        ),
        MetricExporter,
    )


def test_setup_exporter_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_exporter({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
