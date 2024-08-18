r"""Contain the base class to implement an evaluator."""

from __future__ import annotations

__all__ = ["BaseEvaluator", "setup_evaluator", "is_evaluator_config"]

import logging
from abc import ABC
from typing import TYPE_CHECKING

from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    from arkas.result import BaseResult

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC, metaclass=AbstractFactory):
    r"""Define the base class to evaluate a DataFrame.

    Example usage:

    ```pycon

    >>> import polars as pl

    ```
    """

    def evaluate(self, data: dict) -> BaseResult:
        r"""Evaluate the results.

        Args:
            data: The data to evaluate.

        Returns:
            The generated results.

        Example usage:

        ```pycon

        >>> import polars as pl

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
