r"""Contain the implementation of a HTML content generator that analyzes
accuracy performances."""

from __future__ import annotations

__all__ = ["AccuracyContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.evaluator2.accuracy import AccuracyEvaluator

if TYPE_CHECKING:
    from arkas.state import AccuracyState


logger = logging.getLogger(__name__)


class AccuracyContentGenerator(BaseSectionContentGenerator):
    r"""Implement a HTML content generator that analyzes accuracy
    performances.

    Args:
        evaluator: The evaluator object to compute the accuracy.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.content import AccuracyContentGenerator
    >>> from arkas.evaluator2 import AccuracyEvaluator
    >>> from arkas.state import AccuracyState
    >>> content = AccuracyContentGenerator(
    ...     AccuracyEvaluator(
    ...         AccuracyState(
    ...             y_true=np.array([1, 0, 0, 1, 1]),
    ...             y_pred=np.array([1, 0, 0, 1, 1]),
    ...             y_true_name="target",
    ...             y_pred_name="pred",
    ...         )
    ...     )
    ... )
    >>> content
    AccuracyContentGenerator(
      (evaluator): AccuracyEvaluator(
          (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
        )
    )

    ```
    """

    def __init__(self, evaluator: AccuracyEvaluator) -> None:
        self._evaluator = evaluator

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"evaluator": self._evaluator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"evaluator": self._evaluator}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._evaluator.equal(other._evaluator, equal_nan=equal_nan)

    def generate_content(self) -> str:
        logger.info("Generating the accuracy content...")
        metrics = self._evaluator.evaluate()
        return Template(create_template()).render(
            {
                "accuracy": f"{metrics.get('accuracy', float('nan')):.4f}",
                "count": f"{metrics.get('count', 0):,}",
                "count_correct": f"{metrics.get('count_correct', 0):,}",
                "count_incorrect": f"{metrics.get('count_incorrect', 0):,}",
                "error": f"{metrics.get('error', float('nan')):.4f}",
                "y_true_name": self._evaluator.state.y_true_name,
                "y_pred_name": self._evaluator.state.y_pred_name,
            }
        )

    @classmethod
    def from_state(cls, state: AccuracyState) -> AccuracyContentGenerator:
        r"""Instantiate a ``AccuracyContentGenerator`` object from a
        state.

        Args:
            state: The state with the data to analyze.

        Returns:
            The instantiated object.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.content import AccuracyContentGenerator
        >>> from arkas.state import AccuracyState
        >>> content = AccuracyContentGenerator.from_state(
        ...     state=AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> content
        AccuracyContentGenerator(
          (evaluator): AccuracyEvaluator(
              (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
            )
        )

        ```
        """
        return cls(AccuracyEvaluator(state))


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.content.accuracy import create_template
    >>> template = create_template()

    ```
    """
    return """<ul>
  <li><b>accuracy</b>: {{accuracy}} ({{count_correct}}/{{count}})</li>
  <li><b>error</b>: {{error}} ({{count_incorrect}}/{{count}})</li>
  <li><b>number of samples</b>: {{count}}</li>
  <li><b>target label column</b>: {{y_true_name}}</li>
  <li><b>predicted label column</b>: {{y_pred_name}}</li>
</ul>
"""
