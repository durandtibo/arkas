r"""Contain a base class that partially implements the lazy computation
logic."""

from __future__ import annotations

__all__ = ["BaseLazyOutput"]

from abc import abstractmethod
from typing import TYPE_CHECKING

from arkas.evaluator2.vanilla import Evaluator
from arkas.hcg.vanilla import ContentGenerator
from arkas.output.base import BaseOutput
from arkas.plotter.vanilla import Plotter

if TYPE_CHECKING:
    from arkas.evaluator2.base import BaseEvaluator
    from arkas.hcg.base import BaseContentGenerator
    from arkas.plotter.base import BasePlotter


class BaseLazyOutput(BaseOutput):
    r"""Define a base class that partially implements the lazy
    computation logic."""

    def get_content_generator(self, lazy: bool = True) -> BaseContentGenerator:
        generator = self._get_content_generator()
        if lazy:
            return generator
        return ContentGenerator(content=generator.generate_body())

    def get_evaluator(self, lazy: bool = True) -> BaseEvaluator:
        evaluator = self._get_evaluator()
        if lazy:
            return evaluator
        return Evaluator(metrics=evaluator.evaluate())

    def get_plotter(self, lazy: bool = True) -> BasePlotter:
        plotter = self._get_plotter()
        if lazy:
            return plotter
        return Plotter(plotter.plot())

    @abstractmethod
    def _get_content_generator(self) -> BaseContentGenerator:
        r"""Get the HTML content generator associated to the output.

        Returns:
            The HTML content generator.
        """

    @abstractmethod
    def _get_evaluator(self) -> BaseEvaluator:
        r"""Get the evaluator associated to the output.

        Returns:
            The evaluator.
        """

    @abstractmethod
    def _get_plotter(self) -> BasePlotter:
        r"""Get the plotter associated to the output.

        Returns:
            The plotter.
        """
