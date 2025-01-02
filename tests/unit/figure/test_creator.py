from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from arkas.figure.creator import BaseFigureCreator, FigureCreatorRegistry

if TYPE_CHECKING:
    from arkas.figure import BaseFigure, BaseFigureConfig


class MyFigureCreator(BaseFigureCreator):

    def create(self, config: BaseFigureConfig) -> BaseFigure:  # noqa: ARG002
        return HtmlFigure("")

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return isinstance(other, self.__class__)


###########################################
#     Tests for FigureCreatorRegistry     #
###########################################


def test_figure_creator_registry_repr() -> None:
    assert repr(FigureCreatorRegistry()).startswith("FigureCreatorRegistry(")


def test_figure_creator_registry_str() -> None:
    assert repr(FigureCreatorRegistry()).startswith("FigureCreatorRegistry(")


def test_figure_creator_registry_add_creator() -> None:
    registry = FigureCreatorRegistry()
    registry.add_creator("b1", MyFigureCreator())
    assert len(registry._registry) == 1


def test_figure_creator_registry_add_creator_exist_ok_false() -> None:
    registry = FigureCreatorRegistry()
    registry.add_creator("b1", MyFigureCreator())
    with pytest.raises(
        RuntimeError, match="A figure creator .* is already registered for the backend 'b1'"
    ):
        registry.add_creator("b1", MyFigureCreator())


def test_figure_creator_registry_add_creator_exist_ok_true() -> None:
    registry = FigureCreatorRegistry()
    registry.add_creator("b1", MyFigureCreator())
    registry.add_creator("b1", MyFigureCreator(), exist_ok=True)
    assert len(registry._registry) == 1


def test_figure_creator_registry_equal_true() -> None:
    assert FigureCreatorRegistry().equal(FigureCreatorRegistry())


def test_figure_creator_registry_equal_false_different_registry() -> None:
    assert not FigureCreatorRegistry().equal(FigureCreatorRegistry({"b1": MyFigureCreator()}))


def test_figure_creator_registry_equal_false_different_type() -> None:
    assert not FigureCreatorRegistry().equal(42)


def test_figure_creator_registry_has_creator_true() -> None:
    assert FigureCreatorRegistry({"b1": MyFigureCreator()}).has_creator("b1")


def test_figure_creator_registry_has_creator_false() -> None:
    assert not FigureCreatorRegistry().has_creator("missing")


def test_figure_creator_registry_find_creator() -> None:
    assert (
        FigureCreatorRegistry({"b1": MyFigureCreator()}).find_creator("b1").equal(MyFigureCreator())
    )


def test_figure_creator_registry_find_creator_missing() -> None:
    registry = FigureCreatorRegistry()
    with pytest.raises(ValueError, match="Incorrect backend: 'missing'"):
        registry.find_creator("missing")
