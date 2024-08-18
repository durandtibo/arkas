from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl
from grizz.ingestor import Ingestor
from objectory import OBJECT_TARGET

from arkas.ingestor import DataFrameIngestor, is_ingestor_config, setup_ingestor

if TYPE_CHECKING:
    import pytest

########################################
#     Tests for is_ingestor_config     #
########################################


def test_is_ingestor_config_true() -> None:
    assert is_ingestor_config({OBJECT_TARGET: "arkas.ingestor.DataFrameIngestor"})


def test_is_ingestor_config_false() -> None:
    assert not is_ingestor_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_ingestor     #
####################################


def test_setup_ingestor_object() -> None:
    generator = DataFrameIngestor(Ingestor(pl.DataFrame()))
    assert setup_ingestor(generator) is generator


def test_setup_ingestor_dict() -> None:
    assert isinstance(
        setup_ingestor(
            {
                OBJECT_TARGET: "arkas.ingestor.DataFrameIngestor",
                "ingestor": {OBJECT_TARGET: "grizz.ingestor.Ingestor", "frame": pl.DataFrame()},
            }
        ),
        DataFrameIngestor,
    )


def test_setup_ingestor_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_ingestor({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
