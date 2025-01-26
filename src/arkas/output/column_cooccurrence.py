r"""Implement the pairwise column co-occurrence output."""

from __future__ import annotations

__all__ = ["ColumnCooccurrenceOutput"]


from arkas.content.column_cooccurrence import ColumnCooccurrenceContentGenerator
from arkas.evaluator2.column_cooccurrence import ColumnCooccurrenceEvaluator
from arkas.output.state import BaseStateOutput
from arkas.state.column_cooccurrence import ColumnCooccurrenceState


class ColumnCooccurrenceOutput(BaseStateOutput[ColumnCooccurrenceState]):
    r"""Implement the pairwise column co-occurrence output.

    Args:
        state: The state with the co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import ColumnCooccurrenceOutput
    >>> from arkas.state import ColumnCooccurrenceState
    >>> output = ColumnCooccurrenceOutput(
    ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ... )
    >>> output
    ColumnCooccurrenceOutput(
      (state): ColumnCooccurrenceState(matrix=(3, 3), figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    ColumnCooccurrenceContentGenerator(
      (state): ColumnCooccurrenceState(matrix=(3, 3), figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    ColumnCooccurrenceEvaluator(
      (state): ColumnCooccurrenceState(matrix=(3, 3), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    def __init__(self, state: ColumnCooccurrenceState) -> None:
        super().__init__(state)
        self._content = ColumnCooccurrenceContentGenerator(self._state)
        self._evaluator = ColumnCooccurrenceEvaluator(state=self._state)

    def _get_content_generator(self) -> ColumnCooccurrenceContentGenerator:
        return self._content

    def _get_evaluator(self) -> ColumnCooccurrenceEvaluator:
        return self._evaluator
