r"""Contain the base class to implement an output exporter."""

from __future__ import annotations

__all__ = ["BaseExporter", "is_exporter_config", "setup_exporter"]

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from coola.equality.comparators import BaseEqualityComparator
from coola.equality.handlers import EqualNanHandler, SameObjectHandler, SameTypeHandler
from coola.equality.testers import EqualityTester
from objectory import AbstractFactory
from objectory.utils import is_object_config

if TYPE_CHECKING:
    from coola.equality import EqualityConfig

    from arkas.output.base import BaseOutput

logger = logging.getLogger(__name__)


class BaseExporter(ABC, metaclass=AbstractFactory):
    r"""Define the base class to export an output object.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from arkas.output import AccuracyOutput
    >>> from arkas.state import AccuracyState
    >>> from arkas.exporter import MetricExporter
    >>> output = AccuracyOutput(
    ...     state=AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     exporter = MetricExporter(path=Path(tmpdir).joinpath("metrics.pkl"))
    ...     print(exporter)
    ...     exporter.export(output)
    ...
    MetricExporter(
      (path): .../metrics.pkl
      (saver): PickleSaver()
      (exist_ok): False
      (show_metrics): False
    )

    ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two exporters are equal or not.

        Args:
            other: The other exporter to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two exporters are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> from arkas.exporter import MetricExporter
        >>> exporter1 = MetricExporter(path="/data/metrics.pkl")
        >>> exporter2 = MetricExporter(path="/data/metrics.pkl")
        >>> exporter3 = MetricExporter(path="/data/metrics.pkl", exist_ok=True)
        >>> exporter1.equal(exporter2)
        True
        >>> exporter1.equal(exporter3)
        False

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
        >>> import numpy as np
        >>> from arkas.output import AccuracyOutput
        >>> from arkas.state import AccuracyState
        >>> from arkas.exporter import MetricExporter
        >>> output = AccuracyOutput(
        ...     state=AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     exporter = MetricExporter(path=Path(tmpdir).joinpath("metrics.pkl"))
        ...     exporter.export(output)
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
    ...         "_target_": "arkas.exporter.MetricExporter",
    ...         "path": "/path/to/data.csv",
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
    ...         "_target_": "arkas.exporter.MetricExporter",
    ...         "path": "/path/to/data.csv",
    ...     }
    ... )
    >>> exporter
    MetricExporter(
      (path): /path/to/data.csv
      (saver): PickleSaver()
      (exist_ok): False
      (show_metrics): False
    )

    ```
    """
    if isinstance(exporter, dict):
        logger.info("Initializing a exporter from its configuration... ")
        exporter = BaseExporter.factory(**exporter)
    if not isinstance(exporter, BaseExporter):
        logger.warning(f"exporter is not a 'BaseExporter' (received: {type(exporter)})")
    return exporter


class ExporterEqualityComparator(BaseEqualityComparator[BaseExporter]):
    r"""Implement an equality comparator for ``BaseExporter``
    objects."""

    def __init__(self) -> None:
        self._handler = SameObjectHandler()
        self._handler.chain(SameTypeHandler()).chain(EqualNanHandler())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__)

    def clone(self) -> ExporterEqualityComparator:
        return self.__class__()

    def equal(self, actual: BaseExporter, expected: Any, config: EqualityConfig) -> bool:
        return self._handler.handle(actual, expected, config=config)


if not EqualityTester.has_comparator(BaseExporter):  # pragma: no cover
    EqualityTester.add_comparator(BaseExporter, ExporterEqualityComparator())
