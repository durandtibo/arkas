r"""Implement the classification accuracy output."""

from __future__ import annotations

__all__ = ["EmptyOutput"]

from typing import Any

from arkas.evaluator2.vanilla import Evaluator
from arkas.hcg.vanilla import ContentGenerator
from arkas.output.base import BaseOutput
from arkas.plotter.vanilla import Plotter


class EmptyOutput(BaseOutput):
    r"""Implement the accuracy output.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import EmptyOutput
    >>> output = EmptyOutput()
    >>> output
    EmptyOutput()
    >>> output.get_evaluator()
    Evaluator(count=0)

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return isinstance(other, self.__class__)

    def get_content_generator(self, lazy: bool = True) -> ContentGenerator:  # noqa: ARG002
        return ContentGenerator()

    def get_evaluator(self, lazy: bool = True) -> Evaluator:  # noqa: ARG002
        return Evaluator()

    def get_plotter(self, lazy: bool = True) -> Plotter:  # noqa: ARG002
        return Plotter()
