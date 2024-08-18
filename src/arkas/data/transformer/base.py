r"""Contain the base class to implement a data transformer."""

from __future__ import annotations

__all__ = ["BaseTransformer", "is_transformer_config", "setup_transformer"]

import logging
from abc import ABC

from objectory import AbstractFactory
from objectory.utils import is_object_config

logger = logging.getLogger(__name__)


class BaseTransformer(ABC, metaclass=AbstractFactory):
    r"""Define the base class to transform data.

    Example usage:

    ```pycon

    >>> import polars as pl

    ```
    """

    def transform(self, data: dict) -> dict:
        r"""Transform the data.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.

        Example usage:

        ```pycon

        >>> import polars as pl

        ```
        """


def is_transformer_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseTransformer``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseTransformer`` object.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import is_transformer_config
    >>> is_transformer_config(
    ...     {
    ...         "_target_": "grizz.transformer.Cast",
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseTransformer)


def setup_transformer(
    transformer: BaseTransformer | dict,
) -> BaseTransformer:
    r"""Set up a data transformer.

    The transformer is instantiated from its configuration
    by using the ``BaseTransformer`` factory function.

    Args:
        transformer: A data transformer or its configuration.

    Returns:
        An instantiated transformer.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from grizz.transformer import setup_transformer
    >>> transformer = setup_transformer(
    ...     {
    ...         "_target_": "grizz.transformer.Cast",
    ...         "columns": ("col1", "col3"),
    ...         "dtype": pl.Int32,
    ...     }
    ... )
    >>> transformer
    CastTransformer(columns=('col1', 'col3'), dtype=Int32, ignore_missing=False)

    ```
    """
    if isinstance(transformer, dict):
        logger.info("Initializing a DataFrame transformer from its configuration... ")
        transformer = BaseTransformer.factory(**transformer)
    if not isinstance(transformer, BaseTransformer):
        logger.warning(f"transformer is not a `BaseTransformer` (received: {type(transformer)})")
    return transformer
