r"""Contain the base class to implement a runner."""

from __future__ import annotations

__all__ = ["BaseRunner", "is_runner_config", "setup_runner"]

import logging
from abc import ABC, abstractmethod
from typing import Any

from objectory import AbstractFactory
from objectory.utils import is_object_config

logger = logging.getLogger(__name__)


class BaseRunner(ABC, metaclass=AbstractFactory):
    r"""Define the base class to implement a runner.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> import polars as pl
    >>> from pathlib import Path
    >>> from iden.io import PickleSaver
    >>> from grizz.ingestor import Ingestor
    >>> from grizz.transformer import SequentialTransformer
    >>> from arkas.evaluator import AccuracyEvaluator
    >>> from arkas.runner import EvaluationRunner
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("metrics.pkl")
    ...     runner = EvaluationRunner(
    ...         ingestor=Ingestor(
    ...             pl.DataFrame(
    ...                 {
    ...                     "pred": [3, 2, 0, 1, 0],
    ...                     "target": [3, 2, 0, 1, 0],
    ...                 }
    ...             )
    ...         ),
    ...         transformer=SequentialTransformer(transformers=[]),
    ...         evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
    ...         saver=PickleSaver(),
    ...         path=path,
    ...     )
    ...     print(runner)
    ...     runner.run()
    ...
    EvaluationRunner(
      (ingestor): Ingestor(shape=(5, 2))
      (transformer): SequentialTransformer()
      (evaluator): AccuracyEvaluator(y_true='target', y_pred='pred', drop_nulls=True, nan_policy='propagate')
      (saver): PickleSaver(protocol=5)
      (path): .../metrics.pkl
      (show_metrics): True
    )

    ```
    """

    @abstractmethod
    def run(self) -> Any:
        r"""Execute the logic of the runner.

        Returns:
            Any artifact of the runner

        Example usage:

        ```pycon

        >>> import tempfile
        >>> import polars as pl
        >>> from pathlib import Path
        >>> from iden.io import PickleSaver
        >>> from grizz.ingestor import Ingestor
        >>> from grizz.transformer import SequentialTransformer
        >>> from arkas.evaluator import AccuracyEvaluator
        >>> from arkas.runner import EvaluationRunner
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     path = Path(tmpdir).joinpath("metrics.pkl")
        ...     runner = EvaluationRunner(
        ...         ingestor=Ingestor(
        ...             pl.DataFrame(
        ...                 {
        ...                     "pred": [3, 2, 0, 1, 0],
        ...                     "target": [3, 2, 0, 1, 0],
        ...                 }
        ...             )
        ...         ),
        ...         transformer=SequentialTransformer(transformers=[]),
        ...         evaluator=AccuracyEvaluator(y_true="target", y_pred="pred"),
        ...         saver=PickleSaver(),
        ...         path=path,
        ...     )
        ...     runner.run()
        ...

        ```
        """


def is_runner_config(config: dict) -> bool:
    r"""Indicate if the input configuration is a configuration for a
    ``BaseRunner``.

    This function only checks if the value of the key  ``_target_``
    is valid. It does not check the other values. If ``_target_``
    indicates a function, the returned type hint is used to check
    the class.

    Args:
        config: The configuration to check.

    Returns:
        ``True`` if the input configuration is a configuration
            for a ``BaseRunner`` object.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.runner import is_runner_config
    >>> is_runner_config(
    ...     {
    ...         "_target_": "arkas.runner.EvaluationRunner",
    ...         "ingestor": {
    ...             "_target_": "grizz.ingestor.Ingestor",
    ...             "frame": pl.DataFrame(
    ...                 {
    ...                     "pred": [3, 2, 0, 1, 0],
    ...                     "target": [3, 2, 0, 1, 0],
    ...                 }
    ...             ),
    ...         },
    ...         "transformer": {"_target_": "grizz.transformer.DropDuplicate"},
    ...         "evaluator": {
    ...             "_target_": "arkas.evaluator.AccuracyEvaluator",
    ...             "y_true": "target",
    ...             "y_pred": "pred",
    ...         },
    ...         "saver": {"_target_": "iden.io.PickleSaver"},
    ...         "path": "/tmp/data/metrics.pkl",
    ...     }
    ... )
    True

    ```
    """
    return is_object_config(config, BaseRunner)


def setup_runner(
    runner: BaseRunner | dict,
) -> BaseRunner:
    r"""Set up a runner.

    The runner is instantiated from its configuration
    by using the ``BaseRunner`` factory function.

    Args:
        runner: Specifies a runner or its configuration.

    Returns:
        An instantiated runner.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.runner import setup_runner
    >>> runner = setup_runner(
    ...     {
    ...         "_target_": "arkas.runner.EvaluationRunner",
    ...         "ingestor": {
    ...             "_target_": "grizz.ingestor.Ingestor",
    ...             "frame": pl.DataFrame(
    ...                 {
    ...                     "pred": [3, 2, 0, 1, 0],
    ...                     "target": [3, 2, 0, 1, 0],
    ...                 }
    ...             ),
    ...         },
    ...         "transformer": {"_target_": "grizz.transformer.DropDuplicate"},
    ...         "evaluator": {
    ...             "_target_": "arkas.evaluator.AccuracyEvaluator",
    ...             "y_true": "target",
    ...             "y_pred": "pred",
    ...         },
    ...         "saver": {"_target_": "iden.io.PickleSaver"},
    ...         "path": "/tmp/data/metrics.pkl",
    ...     }
    ... )
    >>> runner
    EvaluationRunner(
      (ingestor): Ingestor(shape=(5, 2))
      (transformer): DropDuplicateTransformer(columns=None, ignore_missing=False)
      (evaluator): AccuracyEvaluator(y_true='target', y_pred='pred', drop_nulls=True, nan_policy='propagate')
      (saver): PickleSaver(protocol=5)
      (path): .../metrics.pkl
      (show_metrics): True
    )

    ```
    """
    if isinstance(runner, dict):
        logger.info("Initializing a runner from its configuration... ")
        runner = BaseRunner.factory(**runner)
    if not isinstance(runner, BaseRunner):
        logger.warning(f"runner is not a `BaseRunner` (received: {type(runner)})")
    return runner
