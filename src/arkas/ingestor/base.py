r"""Contain the base class to implement a data ingestor."""

from __future__ import annotations

__all__ = ["BaseIngestor", "setup_ingestor", "is_ingestor_config"]

import logging
from abc import ABC

from objectory import AbstractFactory
from objectory.utils import is_object_config

logger = logging.getLogger(__name__)


class BaseIngestor(ABC, metaclass=AbstractFactory):
    r"""Define the base class to ingest data.

    Example usage:

    ```pycon

    >>> import polars as pl

    ```
    """

    def ingest(self) -> dict:
        r"""Ingest the data.

        Returns:
            The ingested data.

        Example usage:

        ```pycon

        >>> import polars as pl

        ```
        """


def is_ingestor_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseIngestor``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseIngestor`` object.

    Example usage:

    ```pycon

    >>> from arkas.ingestor import is_ingestor_config
    >>> is_ingestor_config({"_target_": "arkas.ingestor.DataFrameIngestor"})
    True

    ```
    """
    return is_object_config(config, BaseIngestor)


def setup_ingestor(
    ingestor: BaseIngestor | dict,
) -> BaseIngestor:
    r"""Set up an ingestor.

    The ingestor is instantiated from its configuration
    by using the ``BaseIngestor`` factory function.

    Args:
        ingestor: Specifies an ingestor or its configuration.

    Returns:
        An instantiated ingestor.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.ingestor import setup_ingestor
    >>> from grizz.ingestor import Ingestor
    >>> ingestor = setup_ingestor(
    ...     {
    ...         "_target_": "arkas.ingestor.DataFrameIngestor",
    ...         "ingestor": Ingestor(pl.DataFrame()),
    ...     }
    ... )
    >>> ingestor
    DataFrameIngestor(
      (ingestor): Ingestor(shape=(0, 0))
      (out_key): frame
    )

    ```
    """
    if isinstance(ingestor, dict):
        logger.info("Initializing an ingestor from its configuration... ")
        ingestor = BaseIngestor.factory(**ingestor)
    if not isinstance(ingestor, BaseIngestor):
        logger.warning(f"ingestor is not a `BaseIngestor` (received: {type(ingestor)})")
    return ingestor
