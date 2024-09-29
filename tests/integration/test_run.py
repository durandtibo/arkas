from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

import pytest

from arkas.testing import hydra_available

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def run_bash_command(cmd: str) -> None:
    r"""Execute a bash command.

    Args:
        cmd: The command to run.
    """
    logger.info(f"execute the following command: {cmd}")
    subprocess.run(cmd.split(), check=True)  # noqa: S603


@hydra_available
def test_run_cli_successful(tmp_path: Path) -> None:
    path = tmp_path.joinpath("successful")
    run_bash_command(
        f"python -m arkas.cli.run -cd=conf/demo -cn=toy_binary_classification exp_dir={path}"
    )
    assert path.joinpath("metrics.pkl").is_file()


@hydra_available
def test_run_cli_error(tmp_path: Path) -> None:
    path = tmp_path.joinpath("unsuccessful")
    with pytest.raises(subprocess.CalledProcessError):
        run_bash_command(
            r"python -m arkas.cli.run -cd=conf/demo -cn=toy_binary_classification "
            rf"exp_dir={path} ingestor.path={tmp_path}"
        )
    assert not tmp_path.joinpath("metrics.pkl").is_file()
