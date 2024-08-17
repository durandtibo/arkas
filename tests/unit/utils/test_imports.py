from __future__ import annotations

from unittest.mock import patch

import pytest

from arkas.utils.imports import (
    check_colorlog,
    check_markdown,
    colorlog_available,
    is_colorlog_available,
    is_markdown_available,
    markdown_available,
)


def my_function(n: int = 0) -> int:
    return 42 + n


####################
#     colorlog     #
####################


def test_check_colorlog_with_package() -> None:
    with patch("arkas.utils.imports.is_colorlog_available", lambda: True):
        check_colorlog()


def test_check_colorlog_without_package() -> None:
    with (
        patch("arkas.utils.imports.is_colorlog_available", lambda: False),
        pytest.raises(RuntimeError, match="`colorlog` package is required but not installed."),
    ):
        check_colorlog()


def test_is_colorlog_available() -> None:
    assert isinstance(is_colorlog_available(), bool)


def test_colorlog_available_with_package() -> None:
    with patch("arkas.utils.imports.is_colorlog_available", lambda: True):
        fn = colorlog_available(my_function)
        assert fn(2) == 44


def test_colorlog_available_without_package() -> None:
    with patch("arkas.utils.imports.is_colorlog_available", lambda: False):
        fn = colorlog_available(my_function)
        assert fn(2) is None


def test_colorlog_available_decorator_with_package() -> None:
    with patch("arkas.utils.imports.is_colorlog_available", lambda: True):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_colorlog_available_decorator_without_package() -> None:
    with patch("arkas.utils.imports.is_colorlog_available", lambda: False):

        @colorlog_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None


####################
#     markdown     #
####################


def test_check_markdown_with_package() -> None:
    with patch("arkas.utils.imports.is_markdown_available", lambda: True):
        check_markdown()


def test_check_markdown_without_package() -> None:
    with (
        patch("arkas.utils.imports.is_markdown_available", lambda: False),
        pytest.raises(RuntimeError, match="`markdown` package is required but not installed."),
    ):
        check_markdown()


def test_is_markdown_available() -> None:
    assert isinstance(is_markdown_available(), bool)


def test_markdown_available_with_package() -> None:
    with patch("arkas.utils.imports.is_markdown_available", lambda: True):
        fn = markdown_available(my_function)
        assert fn(2) == 44


def test_markdown_available_without_package() -> None:
    with patch("arkas.utils.imports.is_markdown_available", lambda: False):
        fn = markdown_available(my_function)
        assert fn(2) is None


def test_markdown_available_decorator_with_package() -> None:
    with patch("arkas.utils.imports.is_markdown_available", lambda: True):

        @markdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) == 44


def test_markdown_available_decorator_without_package() -> None:
    with patch("arkas.utils.imports.is_markdown_available", lambda: False):

        @markdown_available
        def fn(n: int = 0) -> int:
            return 42 + n

        assert fn(2) is None
