r"""Contain the implementation of a HTML content generator that analyzes
accuracy performances."""

from __future__ import annotations

__all__ = ["BalancedAccuracyContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.content.section import BaseSectionContentGenerator
from arkas.evaluator2.balanced_accuracy import BalancedAccuracyEvaluator

if TYPE_CHECKING:
    from arkas.state import AccuracyState


logger = logging.getLogger(__name__)


class BalancedAccuracyContentGenerator(BaseSectionContentGenerator):
    r"""Implement a HTML content generator that analyzes balanced
    accuracy performances.

    Args:
        evaluator: The evaluator object to compute the balanced accuracy.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.content import BalancedAccuracyContentGenerator
    >>> from arkas.state import AccuracyState
    >>> generator = BalancedAccuracyContentGenerator(
    ...     BalancedAccuracyEvaluator(
    ...         AccuracyState(
    ...             y_true=np.array([1, 0, 0, 1, 1]),
    ...             y_pred=np.array([1, 0, 0, 1, 1]),
    ...             y_true_name="target",
    ...             y_pred_name="pred",
    ...         )
    ...     )
    ... )
    >>> generator
    BalancedAccuracyContentGenerator(
      (evaluator): BalancedAccuracyEvaluator(
          (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
        )
    )

    ```
    """

    def __init__(self, evaluator: BalancedAccuracyEvaluator) -> None:
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
        logger.info("Generating the balance accuracy content...")
        metrics = self._evaluator.evaluate()
        return Template(create_template()).render(
            {
                "balanced_accuracy": f"{metrics.get('balanced_accuracy', float('nan')):.4f}",
                "count": f"{metrics.get('count', 0):,}",
                "y_true_name": self._evaluator.state.y_true_name,
                "y_pred_name": self._evaluator.state.y_pred_name,
            }
        )

    @classmethod
    def from_state(cls, state: AccuracyState) -> BalancedAccuracyContentGenerator:
        r"""Instantiate a ``BalancedAccuracyContentGenerator`` object
        from a state.

        Args:
            state: The state with the data to analyze.

        Returns:
            The instantiated object.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.content import BalancedAccuracyContentGenerator
        >>> from arkas.state import AccuracyState
        >>> content = BalancedAccuracyContentGenerator.from_state(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> content
        BalancedAccuracyContentGenerator(
          (evaluator): BalancedAccuracyEvaluator(
              (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred', nan_policy='propagate')
            )
        )

        ```
        """
        return cls(BalancedAccuracyEvaluator(state))


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
  <li><b>balanced accuracy</b>: {{balanced_accuracy}}</li>
  <li><b>number of samples</b>: {{count}}</li>
  <li><b>target label column</b>: {{y_true_name}}</li>
  <li><b>predicted label column</b>: {{y_pred_name}}</li>
</ul>
"""
