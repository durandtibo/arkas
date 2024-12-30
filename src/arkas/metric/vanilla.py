r"""Contain the implementation of a simple metric."""

from __future__ import annotations

__all__ = ["EmptyMetric", "Metric"]

from typing import Any

from coola import objects_are_equal

from arkas.metric.base import BaseMetric


class Metric(BaseMetric):
    r"""Implement a simple metric.

    Args:
        metrics: The dictionary of metrics.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.metric import AccuracyMetric
    >>> from arkas.state import AccuracyState
    >>> metric = Metric({"accuracy": 1.0, "total": 42})
    >>> metric
    Metric(count=2)
    >>> metric.evaluate()
    {'accuracy': 1.0, 'total': 42}

    ```
    """

    def __init__(self, metrics: dict | None = None) -> None:
        self._metrics = metrics or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={len(self._metrics):,})"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._metrics, other._metrics, equal_nan=equal_nan)

    def evaluate(self, prefix: str = "", suffix: str = "") -> dict:
        return {f"{prefix}{key}{suffix}": value for key, value in self._metrics.items()}


class EmptyMetric(Metric):
    r"""Implement an empty metric."""

    def __init__(self) -> None:
        super().__init__(metrics={})

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"
