r"""Implement an output to analyze the correlation between columns."""

from __future__ import annotations

__all__ = ["BaseStateOutput"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping

from arkas.output.lazy import BaseLazyOutput

if TYPE_CHECKING:
    from arkas.state.target_dataframe import TargetDataFrameState


class BaseStateOutput(BaseLazyOutput):
    r"""Define a base class to implement an output using a state object.

    Args:
        state: The state containing the data to analyze.
    """

    def __init__(self, state: TargetDataFrameState) -> None:
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)
