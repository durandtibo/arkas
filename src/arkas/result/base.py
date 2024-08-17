r"""Contain the base class to implement a result."""

from __future__ import annotations

__all__ = ["BaseResult"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


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

    @abstractmethod
    def render_html_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        r"""Return the HTML body associated to the result.

        Args:
            number: The result number.
            tags: The tags associated to the result.
            depth: The depth in the report.

        Returns:
            The HTML body associated to the result.
        """

    @abstractmethod
    def render_html_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        r"""Return the HTML table of content (TOC) associated to the
        result.

        Args:
            number: The result number associated to the
                result.
            tags: The tags associated to the result.
            depth: The depth in the report.
            max_depth: The maximum depth to generate in the TOC.

        Returns:
            The HTML table of content associated to the result.
        """
