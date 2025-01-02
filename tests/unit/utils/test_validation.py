from __future__ import annotations

import pytest

from arkas.utils.validation import check_positive

####################################
#     Tests for check_positive     #
####################################


@pytest.mark.parametrize("value", [0, 1, 2])
def test_check_positive_correct(value: float) -> None:
    check_positive("var", value)


@pytest.mark.parametrize("value", [-1, -2])
def test_check_positive_incorrect(value: float) -> None:
    with pytest.raises(ValueError, match=f"Incorrect 'var': {value}. The value must be positive"):
        check_positive("var", value)
