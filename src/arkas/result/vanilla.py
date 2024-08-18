r"""Implement the simple result."""

from __future__ import annotations

__all__ = ["Result"]


from arkas.result.base import BaseResult


class Result(BaseResult):
    r"""Implement a simple result.

    Args:
        metrics: The metrics.
        figures: The figures.

    Example usage:

    ```pycon

    >>> from arkas.result import Result
    >>> result = Result(metrics={"accuracy": 1.0, "count": 42}, figures={})
    >>> result
    Result(metrics=2, figures=0)
    >>> result.compute_metrics()
    {'accuracy': 1.0, 'count': 42}

    ```
    """

    def __init__(self, metrics: dict | None = None, figures: dict | None = None) -> None:
        self._metrics = metrics or {}
        self._figures = figures or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(metrics={len(self._metrics):,}, figures={len(self._figures):,})"

    def compute_metrics(self, prefix: str = "", suffix: str = "") -> dict:
        return {f"{prefix}{key}{suffix}": value for key, value in self._metrics.items()}

    def generate_figures(self, prefix: str = "", suffix: str = "") -> dict:
        return {f"{prefix}{key}{suffix}": value for key, value in self._figures.items()}
