r"""Contain the base class to implement a section."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod


class BaseResult(ABC):
    r"""Define the base class to manage results."""

    @abstractmethod
    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict:
        r"""Return the metrics associated to the result.

        Args:
            prefix: The key prefix in the returned dictionary.
            suffix: The key suffix in the returned dictionary.

        Returns:
            The metrics.
        """

    @abstractmethod
    def generate_figures(self, prefix: str = "", suffix: str = "") -> dict:
        r"""Return the figures associated to the result.

        Args:
            prefix: The key prefix in the returned dictionary.
            suffix: The key suffix in the returned dictionary.

        Returns:
            The figures.
        """
