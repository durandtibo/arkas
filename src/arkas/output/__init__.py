r"""Contain data outputs."""

from __future__ import annotations

__all__ = ["AccuracyOutput", "BaseLazyOutput", "BaseOutput", "EmptyOutput", "Output"]

from arkas.output.accuracy import AccuracyOutput
from arkas.output.base import BaseOutput
from arkas.output.empty import EmptyOutput
from arkas.output.lazy import BaseLazyOutput
from arkas.output.vanilla import Output
