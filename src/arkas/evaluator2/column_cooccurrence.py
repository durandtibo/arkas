r"""Implement the pairwise column co-occurrence evaluator."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceEvaluator"]

from typing import TYPE_CHECKING

from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.state.column_cooccurrence import ColumnCooccurrenceState

if TYPE_CHECKING:
    import numpy as np


class ColumnCooccurrenceEvaluator(BaseStateCachedEvaluator[ColumnCooccurrenceState]):
    r"""Implement the pairwise column co-occurrence evaluator.

    Args:
        state: The state with the co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.evaluator2 import ColumnCooccurrenceEvaluator
    >>> from arkas.state import ColumnCooccurrenceState
    >>> evaluator = ColumnCooccurrenceEvaluator(
    ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ... )
    >>> evaluator
    ColumnCooccurrenceEvaluator(
      (state): ColumnCooccurrenceState(matrix=(3, 3), figure_config=MatplotlibFigureConfig())
    )
    >>> evaluator.evaluate()
    {'column_cooccurrence': array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])}

    ```
    """

    def _evaluate(self) -> dict[str, np.ndarray]:
        return {"column_cooccurrence": self._state.matrix}
