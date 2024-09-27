r"""Contain the entrypoint to run evaluations or analyses."""

from __future__ import annotations

__all__ = ["main", "main_cli"]

import logging
from typing import Any

import hydra
from hya import register_resolvers
from iden.utils.time import timeblock
from omegaconf import DictConfig, OmegaConf

from arkas.cli.utils import log_run_info
from arkas.runner import setup_runner

logger = logging.getLogger(__name__)


def main(config: dict[str, Any]) -> None:
    r"""Initialize a runner given its configuration and execute its
    logic.

    Args:
        config: The dictionary with the configuration of the runner.
            This dictionary has to have a key ``'runner'``.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.cli.run import main
    >>> main(
    ...     {
    ...         "runner": {
    ...             "_target_": "arkas.runner.EvaluationRunner",
    ...             "ingestor": {
    ...                 "_target_": "grizz.ingestor.Ingestor",
    ...                 "frame": pl.DataFrame(
    ...                     {
    ...                         "pred": [3, 2, 0, 1, 0],
    ...                         "target": [3, 2, 0, 1, 0],
    ...                     }
    ...                 ),
    ...             },
    ...             "evaluator": {
    ...                 "_target_": "arkas.evaluator.AccuracyEvaluator",
    ...                 "y_true": "target",
    ...                 "y_pred": "pred",
    ...             },
    ...             "saver": {"_target_": "iden.io.PickleSaver"},
    ...             "path": "/tmp/data/metrics.pkl",
    ...         }
    ...     }
    ... )

    ```
    """
    with timeblock("Total time of the run: {time}"):
        logger.info("Creating runner...")
        runner = setup_runner(config["runner"])
        logger.info(f"runner:\n{runner}")
        logger.info("Start to execute the logic of the runner")
        runner.run()
        logger.info("End of the run")


@hydra.main(config_path=None, version_base=None)
def main_cli(config: DictConfig) -> None:
    r"""Define the CLI entrypoint to run an experiment.

    Please check the Hydra documentation to learn how Hydra works:
    https://hydra.cc/

    Args:
        config: The dictionary with the configuration of the runner.
            This dictionary has to have a key ``'runner'``.
    """
    log_run_info(config)
    register_resolvers()
    main(OmegaConf.to_container(config, resolve=True))


if __name__ == "__main__":  # pragma: no cover
    main_cli()
