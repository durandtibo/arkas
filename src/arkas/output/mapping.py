r"""Implement a output that combines a mapping of output objects into a
single output object."""

from __future__ import annotations

__all__ = ["MappingOutput"]

from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import str_indent, str_mapping

from arkas.output import BaseOutput

if TYPE_CHECKING:
    from collections.abc import Mapping


class MappingOutput(BaseOutput):
    r"""Implement an output that combines a mapping of output objects into
    a single output object.

    Args:
        outputs: The mapping of output objects to combine.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import MappingOutput, Output
    >>> output = MappingOutput(
    ...     {
    ...         "class1": Output(metrics={"accuracy": 62.0, "count": 42}),
    ...         "class2": Output(metrics={"accuracy": 42.0, "count": 42}),
    ...     }
    ... )
    >>> output
    MappingOutput(count=2)
    >>> output.compute_metrics()
    {'class1': {'accuracy': 62.0, 'count': 42}, 'class2': {'accuracy': 42.0, 'count': 42}}

    ```
    """

    def __init__(self, outputs: Mapping[str, BaseOutput]) -> None:
        self._outputs = outputs

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(count={len(self._outputs):,})"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self._outputs))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self._outputs, other._outputs, equal_nan=equal_nan)

