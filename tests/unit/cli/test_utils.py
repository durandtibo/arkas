from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import DictConfig, OmegaConf

from arkas.cli.utils import get_original_cwd, log_run_info
from arkas.testing import hydra_available, omegaconf_available

######################################
#     Tests for get_original_cwd     #
######################################


def test_get_original_cwd_no_hydra() -> None:
    with patch("arkas.utils.imports.is_hydra_available", lambda: False):
        assert get_original_cwd() == Path.cwd()


def test_get_original_cwd_with_hydra() -> None:
    with (
        patch("arkas.utils.imports.is_hydra_available", lambda: True),
        patch("hydra.core.hydra_config.HydraConfig.initialized", lambda: False),
    ):
        assert get_original_cwd() == Path.cwd()


@hydra_available
def test_get_original_cwd_with_hydra_initialized(tmp_path: Path) -> None:
    with (
        patch("hydra.core.hydra_config.HydraConfig.initialized", lambda: True),
        patch("hydra.utils.get_original_cwd", lambda: tmp_path.as_posix()),
    ):
        assert get_original_cwd() == tmp_path


##################################
#     Tests for log_run_info     #
##################################


@pytest.fixture
def config() -> DictConfig:
    return OmegaConf.create({"k": "v", "list": [1, {"a": "1", "b": "2"}]})


@omegaconf_available
@patch("arkas.cli.utils.get_original_cwd", lambda: "/my/path")
def test_log_run_info(caplog: pytest.LogCaptureFixture, config: DictConfig) -> None:
    caplog.set_level(logging.INFO)
    log_run_info(config)
    assert caplog.messages[-1].startswith("Config:")
