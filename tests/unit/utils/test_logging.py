from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from arkas.testing import colorlog_available
from arkas.utils.logging import configure_logging


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    logging.basicConfig()


#######################################
#     Tests for configure_logging     #
#######################################


@colorlog_available
def test_configure_logging() -> None:
    configure_logging()


def test_configure_logging_without_colorlog() -> None:
    with patch("arkas.utils.logging.is_colorlog_available", lambda: False):
        configure_logging()


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING, logging.ERROR])
def test_configure_logging_level(level: int) -> None:
    with patch("arkas.utils.logging.logging.basicConfig") as bc:
        configure_logging(level)
        assert bc.call_args.kwargs["level"] == level
