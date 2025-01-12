from __future__ import annotations

from arkas.utils.style import get_tab_number_style


def test_get_tab_number_style() -> None:
    assert isinstance(get_tab_number_style(), str)
