from __future__ import annotations

from typing import Any
from unittest.mock import patch

from objectory import OBJECT_TARGET

from arkas.cli.run import main, main_cli
from arkas.runner import BaseRunner
from arkas.testing import omegaconf_available
from arkas.utils.imports import is_omegaconf_available

if is_omegaconf_available():
    from omegaconf import OmegaConf


class FakeRunner(BaseRunner):
    r"""Define a fake runner to test the runner instantiation."""

    def run(self) -> Any:
        pass


def test_main() -> None:
    main({"runner": {OBJECT_TARGET: "FakeRunner"}})


def test_main_factory_call() -> None:
    with patch("arkas.runner.base.BaseRunner.factory") as factory_mock:
        main({"runner": {OBJECT_TARGET: "FakeRunner", "engine": "ABC"}})
        factory_mock.assert_called_with(_target_="FakeRunner", engine="ABC")


@omegaconf_available
def test_main_cli_factory_call() -> None:
    with patch("arkas.runner.base.BaseRunner.factory") as factory_mock:
        main_cli(OmegaConf.create({"runner": {OBJECT_TARGET: "FakeRunner", "engine": "ABC"}}))
        factory_mock.assert_called_with(_target_="FakeRunner", engine="ABC")
    assert OmegaConf.has_resolver("hya.add")
    assert OmegaConf.has_resolver("hya.mul")
