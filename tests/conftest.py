from __future__ import annotations

__all__ = ["sklearn_greater_equal_1_4"]

import operator

import pytest
from feu import compare_version
from matplotlib import pyplot as plt

SKLEARN_GREATER_EQUAL_1_4 = compare_version("scikit-learn", operator.ge, "1.4.0")

sklearn_greater_equal_1_4 = pytest.mark.skipif(
    not SKLEARN_GREATER_EQUAL_1_4, reason="Requires sklearn>=1.4.0"
)


@pytest.fixture(autouse=True)
def _close_plt_figure() -> None:
    plt.close()
