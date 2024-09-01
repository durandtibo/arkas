r"""Contain the base class to implement a runner."""

from __future__ import annotations

__all__ = ["BaseRunner"]

from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory


class BaseRunner(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a runner."""

    @abstractmethod
    def run(self) -> Any:
        r"""Execute the logic of the runner.

        Returns:
            Any artifact of the runner
        """
