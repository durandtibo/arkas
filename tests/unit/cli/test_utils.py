from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from arkas.cli.utils import get_original_cwd, log_run_info
from arkas.testing import hydra_available, omegaconf_available
from arkas.utils.imports import is_omegaconf_available

if TYPE_CHECKING:
    import pytest

if is_omegaconf_available():
    from omegaconf import OmegaConf

######################################
#     Tests for get_original_cwd     #
######################################


def test_get_original_cwd_no_hydra() -> None:
    with patch("arkas.cli.utils.is_hydra_available", lambda: False):
        assert get_original_cwd() == Path.cwd()


@hydra_available
def test_get_original_cwd_with_hydra(tmp_path: Path) -> None:
    with (
        patch("hydra.core.hydra_config.HydraConfig.initialized", lambda: True),
        patch("hydra.utils.get_original_cwd", lambda: tmp_path.as_posix()),
    ):
        assert get_original_cwd() == tmp_path


##################################
#     Tests for log_run_info     #
##################################


@omegaconf_available
@patch("arkas.cli.utils.get_original_cwd", lambda: "/my/path")
def test_log_run_info(caplog: pytest.LogCaptureFixture) -> None:
    config = OmegaConf.create({"k": "v", "list": [1, {"a": "1", "b": "2"}]})
    with caplog.at_level(logging.INFO):
        log_run_info(config)
        assert caplog.messages[-1].startswith("Config:")
