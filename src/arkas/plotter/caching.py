r"""Define the base class to implement a plotter that caches the figures
after the first generation."""

from __future__ import annotations

__all__ = ["BaseCachedPlotter", "BaseStateCachedPlotter"]

from abc import abstractmethod
from typing import Any, Generic, TypeVar

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.plotter.base import BasePlotter
from arkas.plotter.vanilla import Plotter
from arkas.state.base import BaseState

T = TypeVar("T", bound=BaseState)


class BaseCachedPlotter(BasePlotter):
    r"""Define the base class to implement a plotter that caches the
    figures after the first generation."""

    def __init__(self) -> None:
        self._cached_figures = None

    def compute(self) -> Plotter:
        return Plotter(figures=self.plot())

    def plot(self, prefix: str = "", suffix: str = "") -> dict:
        if self._cached_figures is None:
            self._cached_figures = self._plot()
        return {f"{prefix}{col}{suffix}": val for col, val in self._cached_figures.items()}

    @abstractmethod
    def _plot(self) -> dict:
        r"""Generate the figures.

        Returns:
            The figures.
        """


class BaseStateCachedPlotter(BaseCachedPlotter, Generic[T]):
    r"""Define the base class to implement a plotter that caches the
    figures after the first generation, and computes the figures from a
    state object.

    Args:
        state: The state with the data.
    """

    def __init__(self, state: T) -> None:
        super().__init__()
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    @property
    def state(self) -> T:
        return self._state

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)
