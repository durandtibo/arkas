r"""Contain the implementation of a simple HTML content generator."""

from __future__ import annotations

__all__ = ["BaseSectionContentGenerator", "create_template"]

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING

from jinja2 import Template

from arkas.content.base import BaseContentGenerator
from arkas.utils.html import GO_TO_TOP, render_toc, tags2id, tags2title, valid_h_tag

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arkas.content.vanilla import ContentGenerator

logger = logging.getLogger(__name__)


class BaseSectionContentGenerator(BaseContentGenerator):
    r"""Define a base class to implement a section content generator.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.content import ContentGenerator
    >>> generator = ContentGenerator("meow")
    >>> generator
    ContentGenerator()
    >>> generator.generate_content()
    'meow'

    ```
    """

    def compute(self) -> ContentGenerator:
        r"""Compute the content and return a new content generator.

        Returns:
            A new content generator with the computed content.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.content import AccuracyContentGenerator
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
        >>> generator2 = generator.compute()
        >>> generator2
        ContentGenerator()

        ```
        """
        # local import to avoid cyclic dependency because ContentGenerator
        # uses this class as base class
        from arkas.content.vanilla import ContentGenerator

        return ContentGenerator(self.generate_content())

    def generate_body(self, number: str = "", tags: Sequence[str] = (), depth: int = 0) -> str:
        return Template(create_template()).render(
            {
                "go_to_top": GO_TO_TOP,
                "id": tags2id(tags),
                "depth": valid_h_tag(depth + 1),
                "title": tags2title(tags),
                "section": number,
                "content": self.generate_content(),
            }
        )

    def generate_toc(
        self, number: str = "", tags: Sequence[str] = (), depth: int = 0, max_depth: int = 1
    ) -> str:
        return render_toc(number=number, tags=tags, depth=depth, max_depth=max_depth)

    @abstractmethod
    def generate_content(self) -> str:
        r"""Return the section content without the tags.

        Returns:
            The content  without the tags.
        """


def create_template() -> str:
    r"""Return the template of the section.

    Returns:
        The section template.

    Example usage:

    ```pycon

    >>> from arkas.content.accuracy import create_template
    >>> template = create_template()

    ```
    """
    return """<h{{depth}} id="{{id}}">{{section}} {{title}} </h{{depth}}>
{{go_to_top}}
<p style="margin-top: 1rem;">
{{content}}
<p style="margin-top: 1rem;">
"""
