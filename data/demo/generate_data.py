# noqa: INP001
r"""Contain functions to generate data."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from arkas.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def generate_toy_binary_classification_data() -> None:
    r"""Generate a toy binary classification data and save it in a
    parquet file."""
    path = Path.cwd().joinpath("toy_binary_classification.parquet")
    logger.info(f"Generating toy binary classification data at {path}...")
    pl.DataFrame(
        {
            "pred": [1, 0, 0, 1, 1],
            "score": [2, -1, 0, 3, 1],
            "target": [1, 0, 0, 1, 1],
        }
    ).write_parquet(path)


def main() -> None:
    r"""Define the main function to generate data."""
    generate_toy_binary_classification_data()


if __name__ == "__main__":
    configure_logging(level=logging.INFO)
    main()
