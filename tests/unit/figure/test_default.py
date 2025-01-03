from __future__ import annotations

from arkas.figure import DefaultFigureConfig

#########################################
#     Tests for DefaultFigureConfig     #
#########################################


def test_default_figure_config_backend() -> None:
    assert DefaultFigureConfig.backend() == "default"


def test_default_figure_config_repr() -> None:
    assert repr(DefaultFigureConfig()) == "DefaultFigureConfig()"


def test_default_figure_config_str() -> None:
    assert str(DefaultFigureConfig()) == "DefaultFigureConfig()"


def test_default_figure_config_equal_true() -> None:
    assert DefaultFigureConfig().equal(DefaultFigureConfig())


def test_default_figure_config_equal_false_different_type() -> None:
    assert not DefaultFigureConfig().equal(42)


def test_default_figure_config_get_args() -> None:
    assert DefaultFigureConfig().get_args() == {}
