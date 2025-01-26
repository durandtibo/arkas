r"""Implement an output to analyze the number of null values per
column."""

from __future__ import annotations

__all__ = ["NullValueOutput"]


from arkas.content.null_value import NullValueContentGenerator
from arkas.evaluator2.vanilla import Evaluator
from arkas.output.state import BaseStateOutput
from arkas.state.null_value import NullValueState


class NullValueOutput(BaseStateOutput[NullValueState]):
    r"""Implement an output to analyze the number of null values per
    column.

    Args:
        state: The state containing the number of null values per
            column.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import NullValueOutput
    >>> from arkas.state import NullValueState
    >>> output = NullValueOutput(
    ...     NullValueState(
    ...         null_count=np.array([0, 1, 2]),
    ...         total_count=np.array([5, 5, 5]),
    ...         columns=["col1", "col2", "col3"],
    ...     )
    ... )
    >>> output
    NullValueOutput(
      (state): NullValueState(num_columns=3, figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_content_generator()
    NullValueContentGenerator(
      (state): NullValueState(num_columns=3, figure_config=MatplotlibFigureConfig())
    )
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __init__(self, state: NullValueState) -> None:
        super().__init__(state)
        self._content = NullValueContentGenerator(self._state)
        self._evaluator = Evaluator()

    def _get_content_generator(self) -> NullValueContentGenerator:
        return self._content

    def _get_evaluator(self) -> Evaluator:
        return self._evaluator
