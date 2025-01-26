r"""Contain the implementation of a HTML content generator that analyzes
the correlation between two columns."""

from __future__ import annotations

__all__ = [
    "CorrelationContentGenerator",
    "create_template",
]

import logging
from typing import TYPE_CHECKING, Any

from coola import objects_are_equal
from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.evaluator2.correlation import CorrelationEvaluator
from arkas.figure.utils import figure2html
from arkas.plotter.correlation import CorrelationPlotter

if TYPE_CHECKING:
    from arkas.state.columns import TwoColumnDataFrameState

logger = logging.getLogger(__name__)


class CorrelationContentGenerator(BaseSectionContentGenerator):
    r"""Implement a content generator that analyzes the correlation
    between two columns.

    Args:
        evaluator: The evaluator that computes correlation.
        plotter: The data correlation plotter.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.content import CorrelationContentGenerator
    >>> from arkas.state import TwoColumnDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...     },
    ... )
    >>> state = TwoColumnDataFrameState(frame, column1="col1", column2="col2")
    >>> content = CorrelationContentGenerator(
    ...     evaluator=CorrelationEvaluator(state),
    ...     plotter=CorrelationPlotter(state),
    ... )
    >>> content
    CorrelationContentGenerator(
      (evaluator): CorrelationEvaluator(
          (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
        )
      (plotter): CorrelationPlotter(
          (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
        )
    )

    ```
    """

    def __init__(self, evaluator: CorrelationEvaluator, plotter: CorrelationPlotter) -> None:
        self._evaluator = evaluator
        self._plotter = plotter

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping(self.get_args()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping(self.get_args()))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return objects_are_equal(self.get_args(), other.get_args(), equal_nan=equal_nan)

    def generate_content(self) -> str:
        xcol, ycol = self._evaluator.state.column1, self._evaluator.state.column2
        logger.info(f"Generating the correlation analysis between {xcol!r} and {ycol!r}...")
        metrics = self._evaluator.evaluate()
        figures = self._plotter.plot()
        return Template(create_template()).render(
            {
                "xcol": str(xcol),
                "ycol": str(ycol),
                "columns": ", ".join(self._evaluator.state.dataframe.columns),
                "count": f"{metrics['count']:,}",
                "pearson_coeff": f"{metrics['pearson_coeff']:.4f}",
                "pearson_pvalue": f"{metrics['pearson_pvalue']:.4f}",
                "spearman_coeff": f"{metrics['spearman_coeff']:.4f}",
                "spearman_pvalue": f"{metrics['spearman_pvalue']:.4f}",
                "figure": figure2html(figures["correlation"], close_fig=True),
            }
        )

    def get_args(self) -> dict:
        return {"evaluator": self._evaluator, "plotter": self._plotter}

    @classmethod
    def from_state(cls, state: TwoColumnDataFrameState) -> CorrelationContentGenerator:
        r"""Instantiate a ``CorrelationContentGenerator`` object from a
        state.

        Args:
            state: The state with the data to analyze.

        Returns:
            The instantiated object.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.content import CorrelationContentGenerator
        >>> from arkas.state import TwoColumnDataFrameState
        >>> frame = pl.DataFrame(
        ...     {
        ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ...     },
        ... )
        >>> content = CorrelationContentGenerator.from_state(
        ...     TwoColumnDataFrameState(frame, column1="col1", column2="col2")
        ... )
        >>> content
        CorrelationContentGenerator(
          (evaluator): CorrelationEvaluator(
              (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
            )
          (plotter): CorrelationPlotter(
              (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
            )
        )

        ```
        """
        return cls(evaluator=CorrelationEvaluator(state), plotter=CorrelationPlotter(state))


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.correlation import create_template
    >>> template = create_template()

    ```
    """
    return """<p style="margin-top: 1rem;">
This section analyzes the correlation between <em>{{xcol}}</em> and <em>{{ycol}}</em>.
The correlation coefficient is a statistical measure of the strength of a
relationship between two variables. Its values can range from -1 to 1.

<ul>
  <li> <b>pearson coefficient</b>: {{pearson_coeff}} </li>
  <li> <b>pearson p-value</b>: {{pearson_pvalue}} </li>
  <li> <b>spearman coefficient</b>: {{spearman_coeff}} </li>
  <li> <b>spearman p-value</b>: {{spearman_pvalue}} </li>
  <li> <b>num samples</b>: {{count}} </li>
</ul>

<p style="margin-top: 1rem;">
The following figure shows the scatter plot between <em>{{xcol}}</em> and <em>{{ycol}}</em>.
</p>
{{figure}}
"""
