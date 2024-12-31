r"""Contain the base class to implement an output exporter."""

from __future__ import annotations

__all__ = ["BaseExporter", "is_exporter_config", "setup_exporter"]

import logging
from abc import ABC, abstractmethod

from objectory import AbstractFactory
from objectory.utils import is_object_config

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arkas.output.base import BaseOutput

logger = logging.getLogger(__name__)


class BaseExporter(ABC, metaclass=AbstractFactory):
    r"""Define the base class to export an output object.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> import polars as pl
    >>> from arkas.evaluator import AccuracyEvaluator
    >>> from grizz.ingestor import Ingestor
    >>> from grizz.transformer import SequentialTransformer
    >>> from arkas.exporter import EvalExporter
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     exporter = EvalExporter(
    ...         ingestor=Ingestor(
    ...             pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})
    ...         ),
    ...         transformer=SequentialTransformer(transformers=[]),
    ...         evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
    ...         report_path=Path(tmpdir).joinpath("report.html"),
    ...     )
    ...     exporter.generate()
    ...

    ```
    """

    @abstractmethod
    def export(self, output: BaseOutput) -> None:
        r"""Export an output.

        Args:
            output: The output object to export.

        Example usage:

        ```pycon

        >>> import tempfile
        >>> from pathlib import Path
        >>> import polars as pl
        >>> from arkas.evaluator import AccuracyEvaluator
        >>> from grizz.ingestor import Ingestor
        >>> from grizz.transformer import SequentialTransformer
        >>> from arkas.exporter import EvalExporter
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     exporter = EvalExporter(
        ...         ingestor=Ingestor(
        ...             pl.DataFrame({"pred": [3, 2, 0, 1, 0, 1], "target": [3, 2, 0, 1, 0, 1]})
        ...         ),
        ...         transformer=SequentialTransformer(transformers=[]),
        ...         evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        ...         report_path=Path(tmpdir).joinpath("report.html"),
        ...     )
        ...     exporter.generate()
        ...

        ```
        """


def is_exporter_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseExporter``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        bool: ``True`` if the input configuration is a configuration
            for a ``BaseExporter`` object.

    Example usage:

    ```pycon

    >>> from arkas.exporter import is_exporter_config
    >>> is_exporter_config(
    ...     {
    ...         "_target_": "arkas.exporter.EvalExporter",
    ...         "ingestor": {
    ...             "_target_": "grizz.ingestor.CsvIngestor",
    ...             "path": "/path/to/data.csv",
    ...         },
    ...         "transformer": {"_target_": "grizz.transformer.DropDuplicate"},
    ...         "evaluator": {
    ...             "_target_": "arkas.evaluator.AccuracyEvaluator",
    ...             "y_true": "target",
    ...             "y_pred": "pred",
    ...         },
    ...         "report_path": "/path/to/report.html",
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseExporter)


def setup_exporter(
    exporter: BaseExporter | dict,
) -> BaseExporter:
    r"""Set up a exporter.

    The exporter is instantiated from its configuration
    by using the ``BaseExporter`` factory function.

    Args:
        exporter: A exporter or its configuration.

    Returns:
        An instantiated exporter.

    Example usage:

    ```pycon

    >>> from arkas.exporter import setup_exporter
    >>> exporter = setup_exporter(
    ...     {
    ...         "_target_": "arkas.exporter.EvalExporter",
    ...         "ingestor": {
    ...             "_target_": "grizz.ingestor.CsvIngestor",
    ...             "path": "/path/to/data.csv",
    ...         },
    ...         "transformer": {"_target_": "grizz.transformer.DropDuplicate"},
    ...         "evaluator": {
    ...             "_target_": "arkas.evaluator.AccuracyEvaluator",
    ...             "y_true": "target",
    ...             "y_pred": "pred",
    ...         },
    ...         "report_path": "/path/to/report.html",
    ...     }
    ... )
    >>> exporter
    EvalExporter(

    ```
    """
    if isinstance(exporter, dict):
        logger.info("Initializing a exporter from its configuration... ")
        exporter = BaseExporter.factory(**exporter)
    if not isinstance(exporter, BaseExporter):
        logger.warning(f"exporter is not a `BaseExporter` (received: {type(exporter)})")
    return exporter
