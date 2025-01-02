r"""Contain the definition of a figure creator and a registry."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "FigureCreatorRegistry"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import str_indent, str_mapping

if TYPE_CHECKING:
    from arkas.figure.base import BaseFigure, BaseFigureConfig


class BaseFigureCreator(ABC):
    r"""Define the base class to implement a figure creator."""

    @abstractmethod
    def create(self, config: BaseFigureConfig) -> BaseFigure:
        """Create a figure.

        Args:
            config: The figure config.

        Returns:
            The created figure.
        """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two figure creators are equal or not.

        Args:
            other: The other object to compare with.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two objects are equal, otherwise ``False``.
        """


class FigureCreatorRegistry:
    """Implement figure creator registry.

    Args:
        registry: The initial registry with the figure creators.
    """

    def __init__(self, registry: dict[str, BaseFigureCreator] | None = None) -> None:
        self._registry = registry or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(\n  {str_indent(str_mapping(self._registry))}\n)"

    def add_creator(self, backend: str, creator: BaseFigureCreator, exist_ok: bool = False) -> None:
        r"""Add a figure creator for a given backend.

        Args:
            backend: The backend for this test.
            creator: The creator used to test the figure
                of the specified type.
            exist_ok: If ``False``, ``RuntimeError`` is raised if the
                backend already exists. This parameter should be
                set to ``True`` to overwrite the creator for a type.

        Raises:
            RuntimeError: if an creator is already registered for the
                backend and ``exist_ok=False``.

        Example usage:

        ```pycon
        >>> from coola.figure.testers import EqualityTester
        >>> from coola.figure.creators import DefaultEqualityComparator
        >>> tester = EqualityTester.local_copy()
        >>> tester.add_creator(str, DefaultEqualityComparator())
        >>> tester.add_creator(str, DefaultEqualityComparator(), exist_ok=True)

        ```
        """
        if backend in self._registry and not exist_ok:
            msg = (
                f"A figure creator ({self._registry[backend]}) is already registered for the "
                f"backend {backend!r}. Please use `exist_ok=True` if you want to overwrite the "
                "creator for this backend"
            )
            raise RuntimeError(msg)
        self._registry[backend] = creator

    def create(self, config: BaseFigureConfig) -> BaseFigure:
        return self.find_creator(config.backend).create(config)

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._registry, other._registry, equal_nan=equal_nan)

    def has_creator(self, backend: str) -> bool:
        r"""Indicate if a figure creator is registered for the given
        backend.

        Args:
            backend: The backend to check.

        Returns:
            ``True`` if a figure creator is registered,
                otherwise ``False``.

        Example usage:

        ```pycon

        >>> from arkas.figure.creator import FigureCreatorRegistry
        >>> registry = FigureCreatorRegistry()
        >>> registry.has_creator("missing")
        False

        ```
        """
        return backend in self._registry

    def find_creator(self, backend: str) -> BaseFigureCreator:
        r"""Find the figure creator associated to a backend.

        Args:
            backend: The backend.

        Returns:
            The figure creator associated to the backend.

        Raises:
            ValueError: if the backend is missing.

        Example usage:

        ```pycon

        >>> from arkas.figure.creator import FigureCreatorRegistry
        >>> registry = FigureCreatorRegistry()
        >>> registry.find_creator("matplotlib")

        ```
        """
        creator = self._registry.get(backend, None)
        if creator is not None:
            return creator
        msg = f"Incorrect backend: {backend!r}"
        raise ValueError(msg)
