r"""Contain the implementation of a HTML content generator that analyzes
accuracy performances."""

from __future__ import annotations

__all__ = ["AccuracyContentGenerator", "create_template"]

import logging
from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from jinja2 import Template

from arkas.evaluator2 import AccuracyEvaluator
from arkas.hcg.base import BaseContentGenerator
from arkas.metric.utils import check_nan_policy
from arkas.section.utils import (
    GO_TO_TOP,
    render_html_toc,
    tags2id,
    tags2title,
    valid_h_tag,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arkas.state import AccuracyState


logger = logging.getLogger(__name__)


class AccuracyContentGenerator(BaseContentGenerator):
    r"""Implement a HTML content generator that analyzes accuracy
    performances.

    Args:
        state: The data structure containing the states.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.hcg import AccuracyContentGenerator
    >>> from arkas.state import AccuracyState
    >>> generator = AccuracyContentGenerator(
    ...     state=AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> generator
    AccuracyContentGenerator(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )

    ```
    """

    def __init__(self, state: AccuracyState, nan_policy: str = "propagate") -> None:
        self._state = state

        check_nan_policy(nan_policy)
        self._nan_policy = nan_policy

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state, "nan_policy": self._nan_policy}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state, "nan_policy": self._nan_policy}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return (
            self._state.equal(other._state, equal_nan=equal_nan)
            and self._nan_policy == other._nan_policy
        )

    def generate_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        logger.info("Generating the accuracy content...")
        metrics = AccuracyEvaluator(self._state, nan_policy=self._nan_policy).evaluate()
        return Template(create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "accuracy": f"{metrics.get('accuracy', float('nan')):.4f}",
                "count": f"{metrics.get('count', 0):,}",
                "count_correct": f"{metrics.get('count_correct', 0):,}",
                "count_incorrect": f"{metrics.get('count_incorrect', 0):,}",
                "error": f"{metrics.get('error', float('nan')):.4f}",
                "y_true_name": self._state.y_true_name,
                "y_pred_name": self._state.y_pred_name,
            }
        )

    def generate_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        return render_html_toc(number=number, tags=tags, depth=depth, max_depth=max_depth)


def create_template() -> str:
    r"""Return the template of the content.

    Returns:
        The content template.

    Example usage:

    ```pycon

    >>> from arkas.hcg.accuracy import create_template
    >>> template = create_template()

    ```
    """
    return """<h{{depth}} id="{{id}}">{{section}} {{title}} </h{{depth}}>

{{go_to_top}}

<p style="margin-top: 1rem;">

<ul>
  <li>column with target labels: {{y_true_name}}</li>
  <li>column with predicted labels: {{y_pred_name}}</li>
  <li>accuracy: {{accuracy}} ({{count_correct}}/{{count}})</li>
  <li>error: {{error}} ({{count_incorrect}}/{{count}})</li>
</ul>

<p style="margin-top: 1rem;">
"""