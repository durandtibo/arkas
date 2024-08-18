r"""Contain the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator", "setup_evaluator", "is_evaluator_config"]

import logging
from abc import ABC
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    import polars as pl

    from arkas.result import BaseResult

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to analyze a DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator import NullValueEvaluator
    >>> evaluator = NullValueEvaluator()
    >>> evaluator
    NullValueEvaluator(figsize=None)
    >>> frame = pl.DataFrame(
    ...     {
    ...         "float": [1.2, 4.2, None, 2.2],
    ...         "int": [None, 1, 0, 1],
    ...         "str": ["A", "B", None, None],
    ...     },
    ...     schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
    ... )
    >>> section = evaluator.analyze(frame)
    >>> section
    NullValueSection(
      (columns): ('float', 'int', 'str')
      (null_count): array([1, 1, 2])
      (total_count): array([4, 4, 4])
      (figsize): None
    )

    ```
    """

    def analyze(self, frame: pl.DataFrame) -> BaseResult:
        r"""Analyze the data in a DataFrame.

        Args:
            frame: The DataFrame with the data to analyze.

        Returns:
            The section report.

        Example usage:

        ```pycon

        >>> import polars as pl
        >>> from arkas.evaluator import NullValueEvaluator
        >>> evaluator = NullValueEvaluator()
        >>> frame = pl.DataFrame(
        ...     {
        ...         "float": [1.2, 4.2, None, 2.2],
        ...         "int": [None, 1, 0, 1],
        ...         "str": ["A", "B", None, None],
        ...     },
        ...     schema={"float": pl.Float64, "int": pl.Int64, "str": pl.String},
        ... )
        >>> section = evaluator.analyze(frame)
        >>> section
        NullValueSection(
          (columns): ('float', 'int', 'str')
          (null_count): array([1, 1, 2])
          (total_count): array([4, 4, 4])
          (figsize): None
        )

        ```
        """


def is_evaluator_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseEvaluator``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseEvaluator`` object.

    Example usage:

    ```pycon

    >>> from arkas.evaluator import is_evaluator_config
    >>> is_evaluator_config({"_target_": "arkas.evaluator.NullValueEvaluator"})
    True

    ```
    """
    return is_object_config(config, BaseEvaluator)


def setup_evaluator(
    evaluator: BaseEvaluator | dict,
) -> BaseEvaluator:
    r"""Set up an evaluator.

    The evaluator is instantiated from its configuration
    by using the ``BaseEvaluator`` factory function.

    Args:
        evaluator: Specifies an evaluator or its configuration.

    Returns:
        An instantiated evaluator.

    Example usage:

    ```pycon

    >>> from arkas.evaluator import setup_evaluator
    >>> evaluator = setup_evaluator({"_target_": "arkas.evaluator.NullValueEvaluator"})
    >>> evaluator
    NullValueEvaluator(figsize=None)

    ```
    """
    if isinstance(evaluator, dict):
        logger.info("Initializing an evaluator from its configuration... ")
        evaluator = BaseEvaluator.factory(**evaluator)
    if not isinstance(evaluator, BaseEvaluator):
        logger.warning(f"evaluator is not a `BaseEvaluator` (received: {type(evaluator)})")
    return evaluator
