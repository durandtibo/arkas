r"""Contain the base class to implement a section."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod


class BaseResult(ABC):
    r"""Define the base class to manage results."""

    @abstractmethod
    def get_metrics(self) -> dict:
        r"""Return the metrics associated to the result.

        Returns:
            The metrics.
        """

    @abstractmethod
    def get_plots(self) -> dict:
        r"""Return the plots associated to the result.

        Returns:
            The plots.
        """
